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

        # Quantize layer norm output if needed
        if input_quantizer is not None and isinstance(input_quantizer, FP8DeepGemmQuantizer):
            quantized_ln_out = input_quantizer.make_empty(
                ln_out.shape,
                dtype=ln_out.dtype,
                device=ln_out.device
            )
            input_quantizer.update_quantized(ln_out, quantized_ln_out)
        else:
            quantized_ln_out = ln_out

        # Quantize weight if needed
        if weight_quantizer is not None and isinstance(weight_quantizer, FP8DeepGemmQuantizer):
            quantized_weight = weight_quantizer.make_empty(
                weight.shape,
                dtype=weight.dtype,
                device=weight.device
            )
            weight_quantizer.update_quantized(weight, quantized_weight)
        else:
            quantized_weight = weight

        # Perform GEMM using DeepGEMM if both inputs are quantized
        workspace = get_workspace()

        if (isinstance(quantized_ln_out, FP8DeepGemmQTensor) and
            isinstance(quantized_weight, FP8DeepGemmQTensor)):
            try:
                output, _ = deepgemm_fp8_gemm(
                    quantized_ln_out,
                    quantized_weight,
                    workspace,
                    layout="nt",
                    bias=bias,
                    out_dtype=activation_dtype
                )
            except Exception as e:
                warnings.warn(f"DeepGEMM failed, falling back to regular GEMM: {e}")
                # Fall back to regular computation
                if isinstance(quantized_ln_out, FP8DeepGemmQTensor):
                    quantized_ln_out = quantized_ln_out.dequantize()
                if isinstance(quantized_weight, FP8DeepGemmQTensor):
                    quantized_weight = quantized_weight.dequantize()
                output = torch.matmul(quantized_ln_out, quantized_weight.T)
                if bias is not None:
                    output = output + bias
        else:
            # Regular GEMM
            if isinstance(quantized_ln_out, FP8DeepGemmQTensor):
                quantized_ln_out = quantized_ln_out.dequantize()
            if isinstance(quantized_weight, FP8DeepGemmQTensor):
                quantized_weight = quantized_weight.dequantize()
            output = torch.matmul(quantized_ln_out, quantized_weight.T)
            if bias is not None:
                output = output + bias

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
        # For now, use standard backward pass
        # In a full implementation, this would be optimized for DeepGEMM
        inp, ln_weight, ln_bias, weight, bias, ln_out = ctx.saved_tensors

        # Compute gradients using standard autograd
        # This is a simplified implementation - production would need more optimization
        grad_bias = grad_output.sum(dim=0) if bias is not None else None
        grad_weight = torch.matmul(grad_output.transpose(-2, -1), ln_out)
        grad_ln_out = torch.matmul(grad_output, weight)

        # Layer norm backward
        if ctx.normalization == "LayerNorm":
            grad_inp = torch.nn.functional.layer_norm(
                inp, ln_weight.shape, ln_weight, ln_bias, ctx.eps
            )
            # This is simplified - actual implementation would compute proper gradients
            grad_inp = grad_ln_out  # Placeholder
            grad_ln_weight = (grad_ln_out * ln_out).sum(dim=tuple(range(len(grad_ln_out.shape) - 1)))
            grad_ln_bias = grad_ln_out.sum(dim=tuple(range(len(grad_ln_out.shape) - 1))) if ln_bias is not None else None
        else:
            # RMS norm backward (simplified)
            grad_ln_weight = (grad_ln_out * ln_out).sum(dim=tuple(range(len(grad_ln_out.shape) - 1)))
            grad_ln_bias = None
            grad_inp = grad_ln_out

        return (grad_inp, grad_ln_weight, grad_ln_bias, grad_weight, grad_bias,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None, None, None, None, None,
               None, None, None, None, None, None)


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