# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormLinear with DeepGEMM optimization"""

from typing import Optional, Union, Callable, Tuple, Dict, Any
import warnings

import torch
import torch.nn as nn
from torch.nn import init

import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from transformer_engine.common.recipe import Recipe
from .base import (
    TransformerEngineBaseModule,
    get_workspace,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ._common import apply_normalization, WeightGradStore, get_module_quantizers
from ..tensor.float8_deepgemm_tensor import FP8DeepGemmQuantizer, FP8DeepGemmQTensor, DEEPGEMM_AVAILABLE
from ..cpp_extensions.deepgemm import deepgemm_fp8_gemm
from ..fp8 import FP8GlobalStateManager
from ..utils import (
    cast_if_needed,
    get_default_init_method,
    init_method_constant,
    requires_grad,
    needs_quantized_gemm,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    in_fp8_activation_recompute_phase,
)
from ..constants import GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing

if DEEPGEMM_AVAILABLE:
    import deep_gemm

__all__ = ["LayerNormLinearDeepGemm"]


class _LayerNormLinearDeepGemm(torch.autograd.Function):
    """LayerNormLinear with DeepGEMM optimization

    This function extends the regular LayerNormLinear with DeepGEMM-optimized
    matrix multiplication operations.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Union[torch.Tensor, None],
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        wgrad_store: WeightGradStore,
        fuse_wgrad_accumulation: bool,
        input_quantizer: Optional[FP8DeepGemmQuantizer],
        weight_quantizer: Optional[FP8DeepGemmQuantizer],
        output_quantizer: Optional[FP8DeepGemmQuantizer],
        grad_output_quantizer: Optional[FP8DeepGemmQuantizer],
        cpu_offloading: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        tensor_parallel: bool,
        sequence_parallel: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        normalization: str,
        return_layernorm_output: bool,
        return_layernorm_output_gathered: bool,
        is_grad_enabled: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        use_deepgemm: bool,
        module,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:

        # Use regular LayerNormLinear if DeepGEMM not available or disabled
        if not use_deepgemm or not DEEPGEMM_AVAILABLE:
            from .layernorm_linear import _LayerNormLinear
            return _LayerNormLinear.forward(
                ctx, inp, ln_weight, ln_bias, weight, bias, eps,
                is_first_microbatch, fp8, fp8_calibration, wgrad_store,
                fuse_wgrad_accumulation, input_quantizer, weight_quantizer,
                output_quantizer, grad_output_quantizer, cpu_offloading,
                activation_dtype, parallel_mode, tensor_parallel, sequence_parallel,
                tp_group, tp_size, normalization, return_layernorm_output,
                return_layernorm_output_gathered, is_grad_enabled,
                fwd_ln_sm_margin, bwd_ln_sm_margin, zero_centered_gamma, module
            )

        # Apply layer normalization
        if normalization == "LayerNorm":
            ln_out = torch.layer_norm(inp, ln_weight.shape, ln_weight, ln_bias, eps)
        elif normalization == "RMSNorm":
            # RMS normalization
            variance = inp.pow(2).mean(-1, keepdim=True)
            ln_out = inp * torch.rsqrt(variance + eps)
            if ln_weight is not None:
                ln_out = ln_out * ln_weight
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

        # Store for backward pass
        ctx.save_for_backward(inp, ln_weight, ln_bias, weight, bias, ln_out)
        ctx.eps = eps
        ctx.normalization = normalization
        ctx.zero_centered_gamma = zero_centered_gamma
        ctx.use_deepgemm = use_deepgemm
        ctx.is_grad_enabled = is_grad_enabled
        ctx.activation_dtype = activation_dtype

        # Reshape for GEMM if needed
        inp_shape = inp.shape
        if ln_out.dim() > 2:
            ln_out = ln_out.view((-1, ln_out.shape[-1]))

        # Check DeepGEMM requirements - raise error if not met
        if not (ln_out.dim() == 2 and
                ln_out.shape[-1] % 128 == 0 and
                weight.shape[-1] % 128 == 0 and
                weight.shape[0] % 128 == 0):
            raise RuntimeError(f"DeepGEMM requirements not met. "
                             f"ln_out: {ln_out.shape} (must be 2D, last dim % 128 == 0), "
                             f"weight: {weight.shape} (both dims % 128 == 0)")

        if not use_deepgemm:
            raise RuntimeError("DeepGEMM is disabled but LayerNormLinearDeepGemm requires DeepGEMM")

        # Quantize layer norm output - must succeed for DeepGEMM
        if input_quantizer is None or not isinstance(input_quantizer, FP8DeepGemmQuantizer):
            raise RuntimeError("LayerNormLinearDeepGemm requires FP8DeepGemmQuantizer for input")

        quantized_ln_out = input_quantizer.make_empty(
            ln_out.shape,
            dtype=ln_out.dtype,
            device=ln_out.device
        )
        input_quantizer.update_quantized(ln_out, quantized_ln_out)

        # Quantize weight - must succeed for DeepGEMM
        if weight_quantizer is None or not isinstance(weight_quantizer, FP8DeepGemmQuantizer):
            raise RuntimeError("LayerNormLinearDeepGemm requires FP8DeepGemmQuantizer for weight")

        quantized_weight = weight_quantizer.make_empty(
            weight.shape,
            dtype=weight.dtype,
            device=weight.device
        )
        weight_quantizer.update_quantized(weight, quantized_weight)

        # Perform GEMM using DeepGEMM - no fallback
        workspace = get_workspace()

        if not (isinstance(quantized_ln_out, FP8DeepGemmQTensor) and
                isinstance(quantized_weight, FP8DeepGemmQTensor)):
            raise RuntimeError(f"Expected FP8DeepGemmQTensor objects, got {type(quantized_ln_out)}, {type(quantized_weight)}")

        # DeepGEMM forward GEMM - must succeed
        output, _ = deepgemm_fp8_gemm(
            quantized_ln_out,
            quantized_weight,
            workspace,
            layout="nt",
            bias=bias,
            out_dtype=activation_dtype
        )
        print(f"DEBUG: Successfully used DeepGEMM for LayerNormLinear forward")

        # Reshape output back to original shape
        output = output.view(inp_shape[:-1] + (output.shape[-1],))

        # Return based on configuration
        if return_layernorm_output:
            if return_layernorm_output_gathered:
                # TODO: Implement gather operation if needed
                return output, ln_out
            else:
                return output, ln_out
        else:
            return output

    @staticmethod
    def backward(ctx, grad_output, grad_ln_output=None):
        """Backward pass using DeepGEMM operations for both dgrad and wgrad"""
        inp, ln_weight, ln_bias, weight, bias, ln_out = ctx.saved_tensors

        # Reshape grad_output for GEMM
        grad_output_shape = grad_output.shape
        grad_output = grad_output.view((-1, grad_output.shape[-1]))
        ln_out = ln_out.view((-1, ln_out.shape[-1]))

        grad_input = grad_weight = grad_bias = grad_ln_weight = grad_ln_bias = None

        # ==========================================
        # Compute dgrad (input gradients) using DeepGEMM
        # ==========================================
        if ctx.needs_input_grad[0]:  # input requires grad
            # Convert grad_output to FP8 for dgrad
            grad_output_quantizer = FP8DeepGemmQuantizer(
                TE_DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
                use_deepgemm_layout=True,
            )
            grad_output_fp8 = grad_output_quantizer.make_empty(
                grad_output.shape, dtype=grad_output.dtype, device=grad_output.device
            )
            grad_output_quantizer.update_quantized(grad_output, grad_output_fp8)

            # Convert weight to FP8 for dgrad - transpose weight for NT layout
            weight_quantizer = FP8DeepGemmQuantizer(
                TE_DType.kFloat8E4M3,
                rowwise=True,
                columnwise=True,
                use_deepgemm_layout=True,
            )
            # Transpose weight to use NT layout instead of NN layout
            weight_transposed = weight.t().contiguous()
            weight_fp8 = weight_quantizer.make_empty(
                weight_transposed.shape, dtype=weight_transposed.dtype, device=weight_transposed.device
            )
            weight_quantizer.update_quantized(weight_transposed, weight_fp8)

            # Create output tensor for dgrad
            dgrad_gemm_out = torch.empty(
                ln_out.shape, dtype=ctx.activation_dtype, device=ln_out.device
            )

            # dgrad: grad_ln_out = grad_output @ weight (NN layout)
            # Restructure as: grad_ln_out = grad_output @ weight_transposed.T (NT layout)
            deep_gemm.fp8_gemm_nt(
                (grad_output_fp8.rowwise_data, grad_output_fp8.rowwise_scale_inv),
                (weight_fp8.columnwise_data, weight_fp8.columnwise_scale_inv),
                dgrad_gemm_out,
                c=None,
                recipe=None
            )
            print(f"DEBUG: Successfully used DeepGEMM fp8_gemm_nt for LayerNormLinear dgrad")

            # Step 2: Backprop through LayerNorm (simplified)
            if ctx.normalization == "LayerNorm":
                grad_input, grad_ln_weight, grad_ln_bias = _layernorm_backward_simple(
                    dgrad_gemm_out, inp, ln_weight, ln_bias, ctx.eps, ctx.zero_centered_gamma
                )
            else:
                raise ValueError(f"Unsupported normalization: {ctx.normalization}")

            # Reshape grad_input back to original input shape
            grad_input = grad_input.view(grad_output_shape[:-1] + (grad_input.shape[-1],))

        # ==========================================
        # Compute wgrad (weight gradients) using DeepGEMM 1D1D - KEY OPTIMIZATION
        # ==========================================
        if ctx.needs_input_grad[3]:  # weight requires grad
            # Determine if we should accumulate into main_grad (Megatron-LM)
            main_grad = None
            if hasattr(weight, 'main_grad') and weight.main_grad is not None:
                main_grad = weight.main_grad.detach()

            # Convert grad_output to FP8 for wgrad - for correct wgrad computation
            # wgrad: grad_weight = grad_output.T @ ln_out
            # For NT layout: A @ B.T where A=grad_output.T, B.T=ln_out, so B=ln_out.T
            grad_output_quantizer_wgrad = FP8DeepGemmQuantizer(
                TE_DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,  # A tensor uses rowwise
                use_deepgemm_layout=True,
            )
            # Transpose grad_output for correct wgrad computation
            grad_output_transposed = grad_output.t().contiguous()
            grad_output_fp8_wgrad = grad_output_quantizer_wgrad.make_empty(
                grad_output_transposed.shape, dtype=grad_output_transposed.dtype, device=grad_output_transposed.device
            )
            grad_output_quantizer_wgrad.update_quantized(grad_output_transposed, grad_output_fp8_wgrad)

            # Convert ln_out to FP8 for wgrad - transpose for NT layout B tensor
            ln_out_quantizer = FP8DeepGemmQuantizer(
                TE_DType.kFloat8E4M3,
                rowwise=True,
                columnwise=True,  # B tensor uses columnwise for NT layout
                use_deepgemm_layout=True,
            )
            # Transpose ln_out for NT layout: grad_output.T @ ln_out.T (which is B.T)
            ln_out_transposed = ln_out.t().contiguous()
            ln_out_fp8 = ln_out_quantizer.make_empty(
                ln_out_transposed.shape, dtype=ln_out_transposed.dtype, device=ln_out_transposed.device
            )
            ln_out_quantizer.update_quantized(ln_out_transposed, ln_out_fp8)

            # Create output tensor for wgrad
            use_accumulation = main_grad is not None
            if use_accumulation:
                # For main_grad accumulation, use bfloat16 output and handle fp32 accumulation in software
                grad_weight_out = torch.empty(
                    weight.shape, dtype=ctx.activation_dtype, device=weight.device  # Use ctx.activation_dtype (bfloat16)
                )
            else:
                grad_weight_out = torch.empty(
                    weight.shape, dtype=ctx.activation_dtype, device=weight.device
                )

            # wgrad: grad_weight = grad_output.T @ ln_out (NT layout)
            # For NT layout: A @ B.T where A=grad_output.T, B.T=ln_out, so B=ln_out.T
            # grad_output.T: [out_features, batch*seq] @ ln_out: [batch*seq, in_features] -> [out_features, in_features]

            print(f"DEBUG: LayerNormLinear wgrad 1D1D - grad_output: {grad_output.shape} -> transposed: {grad_output_transposed.shape}, ln_out: {ln_out.shape} -> transposed: {ln_out_transposed.shape}, expected output: {weight.shape}")

            deep_gemm.fp8_gemm_nt(
                (grad_output_fp8_wgrad.rowwise_data, grad_output_fp8_wgrad.rowwise_scale_inv),  # A tensor
                (ln_out_fp8.columnwise_data, ln_out_fp8.columnwise_scale_inv),  # B tensor
                grad_weight_out,
                c=None,  # Start with basic case, no accumulation
                recipe=None  # Start with basic case, no forced recipe
            )

            if use_accumulation:
                # Accumulate into main_grad in fp32 for precision
                main_grad.add_(grad_weight_out.to(torch.float32))
                print(f"DEBUG: Successfully used DeepGEMM 1D1D fp8_gemm_nt for LayerNormLinear wgrad with fp32 accumulation")
                grad_weight = None  # Don't return grad_weight when accumulating into main_grad
            else:
                print(f"DEBUG: Successfully used DeepGEMM 1D1D fp8_gemm_nt for LayerNormLinear wgrad")
                grad_weight = grad_weight_out

        # ==========================================
        # Compute grad bias
        # ==========================================
        if ctx.needs_input_grad[4] and bias is not None:  # bias requires grad
            grad_bias = grad_output.sum(dim=0)

        return (grad_input, grad_ln_weight, grad_ln_bias, grad_weight, grad_bias,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None)


def _layernorm_backward_simple(grad_output, input, weight, bias, eps, zero_centered_gamma):
    """Simplified LayerNorm backward pass"""
    # This is a basic implementation - in production you'd use optimized TE kernels
    input_shape = input.shape
    input = input.view(-1, input_shape[-1])
    grad_output = grad_output.view(-1, grad_output.shape[-1])

    # Compute layer norm forward values needed for backward
    mean = input.mean(dim=-1, keepdim=True)
    var = input.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    normalized = (input - mean) / std

    # Compute gradients
    grad_weight = (grad_output * normalized).sum(dim=0)
    grad_bias = grad_output.sum(dim=0) if bias is not None else None

    # Input gradient (simplified)
    grad_normalized = grad_output * weight
    grad_input = grad_normalized / std

    return grad_input.view(input_shape), grad_weight, grad_bias


class LayerNormLinearDeepGemm(TransformerEngineBaseModule):
    """LayerNorm followed by Linear layer with DeepGEMM optimization.

    This module combines layer normalization and linear transformation,
    using DeepGEMM's optimized FP8 GEMM kernels when available.

    Example usage:
    ```python
    layer = LayerNormLinearDeepGemm(
        in_features=4096,
        out_features=4096,
        eps=1e-5,
        use_deepgemm=True,
        fp8_dtype=TE_DType.kFloat8E4M3
    )

    input_tensor = torch.randn(32, 4096, device='cuda', dtype=torch.bfloat16)
    output = layer(input_tensor)
    ```
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        bias: bool = True,
        normalization: str = "LayerNorm",
        init_method: Optional[Callable] = None,
        return_layernorm_output: bool = False,
        return_layernorm_output_gathered: bool = False,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
        dtype: Optional[torch.dtype] = None,
        use_deepgemm: bool = True,
        fp8_dtype: TE_DType = TE_DType.kFloat8E4M3,
        block_scaling_dim: int = 2,
        **kwargs
    ):
        """Initialize LayerNormLinearDeepGemm.

        Parameters
        ----------
        in_features : int
            Size of each input sample
        out_features : int
            Size of each output sample
        eps : float, optional
            Layer normalization epsilon, by default 1e-5
        bias : bool, optional
            Whether to use bias in linear layer, by default True
        normalization : str, optional
            Type of normalization ('LayerNorm' or 'RMSNorm'), by default "LayerNorm"
        init_method : Optional[Callable], optional
            Weight initialization method, by default None
        return_layernorm_output : bool, optional
            Whether to return layer norm output, by default False
        return_layernorm_output_gathered : bool, optional
            Whether to return gathered layer norm output, by default False
        zero_centered_gamma : bool, optional
            Whether to use zero-centered gamma, by default False
        device : Union[torch.device, str], optional
            Device for parameters, by default "cuda"
        dtype : Optional[torch.dtype], optional
            Parameter dtype, by default None
        use_deepgemm : bool, optional
            Whether to use DeepGEMM optimization, by default True
        fp8_dtype : TE_DType, optional
            FP8 data type, by default TE_DType.kFloat8E4M3
        block_scaling_dim : int, optional
            Block scaling dimension, by default 2
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.use_bias = bias
        self.normalization = normalization
        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = return_layernorm_output_gathered
        self.zero_centered_gamma = zero_centered_gamma
        self.use_deepgemm = use_deepgemm and DEEPGEMM_AVAILABLE
        self.fp8_dtype = fp8_dtype

        if dtype is None:
            dtype = torch.bfloat16

        # Layer normalization parameters
        if normalization == "LayerNorm":
            self.ln_weight = nn.Parameter(torch.ones(in_features, device=device, dtype=dtype))
            self.ln_bias = nn.Parameter(torch.zeros(in_features, device=device, dtype=dtype)) if bias else None
        elif normalization == "RMSNorm":
            self.ln_weight = nn.Parameter(torch.ones(in_features, device=device, dtype=dtype))
            self.ln_bias = None
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

        # Linear layer parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        if init_method is None:
            init_method = get_default_init_method()
        init_method(self.weight)

        if zero_centered_gamma:
            with torch.no_grad():
                self.ln_weight.zero_()

        # Create quantizers for DeepGEMM
        if self.use_deepgemm:
            self.input_quantizer = FP8DeepGemmQuantizer(
                fp8_dtype=fp8_dtype,
                rowwise=True,
                columnwise=False,
                block_scaling_dim=block_scaling_dim,
                use_deepgemm_layout=True,
            )
            self.weight_quantizer = FP8DeepGemmQuantizer(
                fp8_dtype=fp8_dtype,
                rowwise=True,
                columnwise=True,
                block_scaling_dim=block_scaling_dim,
                use_deepgemm_layout=True,
            )
        else:
            self.input_quantizer = None
            self.weight_quantizer = None

        # Additional quantizers (can be set by recipes)
        self.output_quantizer = None
        self.grad_output_quantizer = None

        # Other configuration
        self.activation_dtype = dtype
        self.cpu_offloading = False
        self.sequence_parallel = False
        self.tensor_parallel = False
        self.tp_group = None
        self.tp_size = 1
        self.parallel_mode = None

        # Weight gradient store
        self.wgrad_store = WeightGradStore()

    def forward(self, inp: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass with DeepGEMM optimization.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor of shape (..., in_features)

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            Output tensor(s) - single tensor or tuple if return_layernorm_output=True
        """
        # Ensure input is contiguous and on correct device/dtype
        if not inp.is_contiguous():
            inp = inp.contiguous()

        # Determine if we're in FP8 mode
        fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
        fp8_calibration = FP8GlobalStateManager.is_fp8_calibration()

        return _LayerNormLinearDeepGemm.apply(
            inp,
            self.ln_weight,
            self.ln_bias,
            self.weight,
            self.bias,
            self.eps,
            None,  # is_first_microbatch
            fp8_enabled,
            fp8_calibration,
            self.wgrad_store,
            False,  # fuse_wgrad_accumulation
            self.input_quantizer,
            self.weight_quantizer,
            self.output_quantizer,
            self.grad_output_quantizer,
            self.cpu_offloading,
            self.activation_dtype,
            self.parallel_mode,
            self.tensor_parallel,
            self.sequence_parallel,
            self.tp_group,
            self.tp_size,
            self.normalization,
            self.return_layernorm_output,
            self.return_layernorm_output_gathered,
            torch.is_grad_enabled(),
            0,  # fwd_ln_sm_margin
            0,  # bwd_ln_sm_margin
            self.zero_centered_gamma,
            self.use_deepgemm,
            self,
        )

    def set_tensor_parallel_group(self, tp_group: dist_group_type) -> None:
        """Set tensor parallel group"""
        self.tp_group = tp_group
        self.tp_size = get_distributed_world_size(tp_group) if tp_group is not None else 1

    def extra_repr(self) -> str:
        """Extra representation for debugging"""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'eps={self.eps}, bias={self.use_bias}, normalization={self.normalization}, '
                f'use_deepgemm={self.use_deepgemm}, fp8_dtype={self.fp8_dtype}')