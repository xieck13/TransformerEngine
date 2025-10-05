# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DeepGEMM-optimized Linear module with full backward pass support"""

from typing import Optional, Union, Dict, Any, Tuple
import warnings
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from transformer_engine_torch import DType as TE_DType
import deep_gemm

from ..tensor.float8_deepgemm_tensor import FP8DeepGemmQuantizer, FP8DeepGemmQTensor
from ..cpp_extensions.deepgemm import deepgemm_fp8_gemm
from ..cpp_extensions.gemm import general_gemm
from ..utils import _empty_tensor
from .base import TransformerEngineBaseModule, get_workspace, _2X_ACC_WGRAD
from ..distributed import CudaRNGStatesTracker


class _LinearDeepGemmFunction(torch.autograd.Function):
    """Custom autograd function for DeepGEMM linear layer with optimized backward pass"""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        quantizer_input: FP8DeepGemmQuantizer,
        quantizer_weight: FP8DeepGemmQuantizer,
        use_deepgemm: bool,
        dtype: torch.dtype,
        accumulate_into_main_grad: bool = False,
    ):
        """Forward pass with DeepGEMM optimization"""

        # Quantize input
        if isinstance(input, FP8DeepGemmQTensor):
            quantized_input = input
        else:
            quantized_input = quantizer_input.make_empty(
                input.shape, dtype=input.dtype, device=input.device
            )
            quantizer_input.update_quantized(input, quantized_input)

        # Quantize weight
        if isinstance(weight, FP8DeepGemmQTensor):
            quantized_weight = weight
        else:
            quantized_weight = quantizer_weight.make_empty(
                weight.shape, dtype=weight.dtype, device=weight.device
            )
            quantizer_weight.update_quantized(weight, quantized_weight)

        # Get workspace
        workspace = get_workspace()

        # Perform forward GEMM using DeepGEMM
        if (isinstance(quantized_input, FP8DeepGemmQTensor) and
            isinstance(quantized_weight, FP8DeepGemmQTensor) and
            use_deepgemm):

            try:
                output, _ = deepgemm_fp8_gemm(
                    quantized_input,
                    quantized_weight,
                    workspace,
                    layout="nt",
                    bias=bias,
                    out_dtype=dtype
                )
            except Exception as e:
                # Fall back to regular operations
                warnings.warn(f"DeepGEMM forward failed: {e}. Falling back to regular GEMM.")
                output = torch.matmul(quantized_input.dequantize(), quantized_weight.dequantize().T)
                if bias is not None:
                    output = output + bias
        else:
            # Fall back to regular operations
            if isinstance(quantized_input, FP8DeepGemmQTensor):
                quantized_input = quantized_input.dequantize()
            if isinstance(quantized_weight, FP8DeepGemmQTensor):
                quantized_weight = quantized_weight.dequantize()

            output = torch.matmul(quantized_input, quantized_weight.T)
            if bias is not None:
                output = output + bias

        # Save for backward
        ctx.save_for_backward(input, weight, bias)  # Save original tensors, not quantized
        ctx.quantizer_input = quantizer_input
        ctx.quantizer_weight = quantizer_weight
        ctx.use_deepgemm = use_deepgemm
        ctx.dtype = dtype
        ctx.accumulate_into_main_grad = accumulate_into_main_grad

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass using DeepGEMM operations for both dgrad and wgrad"""
        input_tensor, weight_tensor, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        # ==========================================
        # Compute grad input (dgrad) using DeepGEMM
        # ==========================================
        if ctx.needs_input_grad[0]:  # input requires grad
            try:
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

                # Convert weight to FP8 for dgrad
                weight_quantizer = FP8DeepGemmQuantizer(
                    TE_DType.kFloat8E4M3,
                    rowwise=True,
                    columnwise=True,
                    use_deepgemm_layout=True,
                )
                weight_fp8 = weight_quantizer.make_empty(
                    weight_tensor.shape, dtype=weight_tensor.dtype, device=weight_tensor.device
                )
                weight_quantizer.update_quantized(weight_tensor, weight_fp8)

                # Create output tensor for dgrad
                grad_input = torch.empty(
                    input_tensor.shape, dtype=ctx.dtype, device=input_tensor.device
                )

                # dgrad: grad_input = grad_output @ weight (NN layout)
                # grad_output: [batch, out_features] @ weight: [out_features, in_features] -> [batch, in_features]
                deep_gemm.fp8_gemm_nn(
                    (grad_output_fp8.rowwise_data, grad_output_fp8.rowwise_scale_inv),
                    (weight_fp8.columnwise_data, weight_fp8.columnwise_scale_inv),
                    grad_input,
                    c=None,
                    recipe=None
                )
                print(f"DEBUG: Using DeepGEMM fp8_gemm_nn for dgrad")

            except Exception as e:
                print(f"DEBUG: DeepGEMM dgrad fallback to general_gemm: {e}")
                # Fallback to general_gemm
                grad_input, *_ = general_gemm(
                    weight_tensor,
                    grad_output,
                    get_workspace(),
                    out_dtype=ctx.dtype,
                    layout="NN",
                    use_split_accumulator=True,
                    grad=True,
                )

        # ==========================================
        # Compute grad weight (wgrad) using DeepGEMM - KEY OPTIMIZATION
        # ==========================================
        if ctx.needs_input_grad[1]:  # weight requires grad
            try:
                # Determine if we should accumulate into main_grad (Megatron-LM)
                accumulate_into_main_grad = ctx.accumulate_into_main_grad
                main_grad = None
                if accumulate_into_main_grad and hasattr(weight_tensor, 'main_grad'):
                    main_grad = weight_tensor.main_grad
                    if main_grad is not None and main_grad.requires_grad:
                        main_grad = main_grad.detach()

                # Convert input to FP8 for wgrad
                input_quantizer = FP8DeepGemmQuantizer(
                    TE_DType.kFloat8E4M3,
                    rowwise=True,
                    columnwise=False,  # Use rowwise for input in wgrad (better for 1D1D)
                    use_deepgemm_layout=True,
                )
                input_fp8 = input_quantizer.make_empty(
                    input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device
                )
                input_quantizer.update_quantized(input_tensor, input_fp8)

                # Convert grad_output to FP8 for wgrad
                grad_output_quantizer = FP8DeepGemmQuantizer(
                    TE_DType.kFloat8E4M3,
                    rowwise=True,
                    columnwise=True,  # Use columnwise for grad_output in wgrad
                    use_deepgemm_layout=True,
                )
                grad_output_fp8 = grad_output_quantizer.make_empty(
                    grad_output.shape, dtype=grad_output.dtype, device=grad_output.device
                )
                grad_output_quantizer.update_quantized(grad_output, grad_output_fp8)

                # Determine output dtype - KEY: Use fp32 for main_grad accumulation
                wgrad_dtype = torch.float32 if (accumulate_into_main_grad and main_grad is not None) else ctx.dtype

                # Create output tensor for wgrad
                if accumulate_into_main_grad and main_grad is not None:
                    grad_weight_out = main_grad
                    use_accumulation = True
                else:
                    grad_weight_out = torch.empty(
                        weight_tensor.shape, dtype=wgrad_dtype, device=weight_tensor.device
                    )
                    use_accumulation = False

                # wgrad: grad_weight = grad_output.T @ input (NT layout)
                # grad_output.T: [out_features, batch] @ input: [batch, in_features] -> [out_features, in_features]
                # CRITICAL: Use 1D1D kernel when possible by setting c=output for accumulation
                deep_gemm.fp8_gemm_nt(
                    (grad_output_fp8.columnwise_data, grad_output_fp8.columnwise_scale_inv),
                    (input_fp8.rowwise_data, input_fp8.rowwise_scale_inv),
                    grad_weight_out,
                    c=grad_weight_out if use_accumulation else None,  # Enable 1D1D kernel for accumulation
                    recipe=(1, 1, 128) if use_accumulation else None,  # Force 1D1D kernel for accumulation
                )

                if use_accumulation:
                    print(f"DEBUG: Using DeepGEMM fp8_gemm_nt for wgrad with fp32 accumulation (1D1D kernel)")
                    grad_weight = None  # Don't return grad_weight when accumulating into main_grad
                else:
                    print(f"DEBUG: Using DeepGEMM fp8_gemm_nt for wgrad")
                    grad_weight = grad_weight_out

            except Exception as e:
                print(f"DEBUG: DeepGEMM wgrad fallback to general_gemm: {e}")
                # Fallback to general_gemm with the key optimization: fp32 accumulation
                accumulate_into_main_grad = ctx.accumulate_into_main_grad
                main_grad = None
                if accumulate_into_main_grad and hasattr(weight_tensor, 'main_grad'):
                    main_grad = weight_tensor.main_grad
                    if main_grad is not None and main_grad.requires_grad:
                        main_grad = main_grad.detach()

                wgrad_dtype = torch.float32 if (accumulate_into_main_grad and main_grad is not None) else ctx.dtype

                if accumulate_into_main_grad and main_grad is not None:
                    grad_weight, *_ = general_gemm(
                        input_tensor,
                        grad_output,
                        get_workspace(),
                        out_dtype=torch.float32,  # Force fp32 for main_grad accumulation
                        layout="NT",
                        accumulate=True,
                        out=main_grad,
                        use_split_accumulator=_2X_ACC_WGRAD,
                        grad=True,
                    )
                    grad_weight = None
                else:
                    grad_weight, *_ = general_gemm(
                        input_tensor,
                        grad_output,
                        get_workspace(),
                        out_dtype=ctx.dtype,
                        layout="NT",
                        accumulate=False,
                        use_split_accumulator=_2X_ACC_WGRAD,
                        grad=True,
                    )

        # ==========================================
        # Compute grad bias
        # ==========================================
        if ctx.needs_input_grad[2] and bias is not None:  # bias requires grad
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class LinearDeepGemm(TransformerEngineBaseModule):
    """DeepGEMM-optimized Linear layer with full backward pass support.

    This module is a drop-in replacement for TE's Linear that uses
    FP8DeepGemmQuantizer to leverage DeepGEMM's optimized kernels for
    both forward and backward passes.

    Key features:
    - fp32 weight gradient accumulation for better precision
    - 1D1D kernel preference for weight gradient accumulation
    - Megatron-LM main_grad compatibility
    - Tensor parallelism support
    - Sequence parallelism support

    Example usage:
    ```python
    # Drop-in replacement for TE Linear
    linear = LinearDeepGemm(
        in_features=4096,
        out_features=4096,
        bias=True,
        accumulate_into_main_grad=False  # Set True for Megatron-LM
    )

    # Forward and backward pass
    input_tensor = torch.randn(32, 4096, device='cuda', dtype=torch.bfloat16)
    output = linear(input_tensor)
    loss = output.sum()
    loss.backward()  # Will use DeepGEMM for both dgrad and wgrad
    ```
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        fp8_dtype: TE_DType = TE_DType.kFloat8E4M3,
        use_deepgemm: bool = True,
        block_scaling_dim: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        sequence_parallel: bool = False,
        rng_state_tracker_function: Optional[callable] = None,
        accumulate_into_main_grad: bool = False,
        **kwargs
    ):
        """Initialize LinearDeepGemm module.

        Parameters
        ----------
        in_features : int
            Size of each input sample
        out_features : int
            Size of each output sample
        bias : bool, optional
            Whether to use bias, by default True
        fp8_dtype : TE_DType, optional
            FP8 data type, by default TE_DType.kFloat8E4M3
        use_deepgemm : bool, optional
            Whether to use DeepGEMM optimization, by default True
        block_scaling_dim : int, optional
            Block scaling dimension (1 or 2), by default 2
        device : Optional[torch.device], optional
            Device to create tensors on, by default None
        dtype : Optional[torch.dtype], optional
            Data type for non-quantized tensors, by default None
        tensor_parallel_mode : Optional[str], optional
            Mode for tensor parallelism ("column" or "row"), by default None
        tensor_parallel_group : Optional[torch.distributed.ProcessGroup], optional
            Process group for tensor parallelism, by default None
        sequence_parallel : bool, optional
            Whether to apply sequence parallelism, by default False
        rng_state_tracker_function : Optional[CudaRNGStatesTracker], optional
            Function that returns CudaRNGStatesTracker for model-parallel weight init, by default None
        accumulate_into_main_grad : bool, optional
            Whether to accumulate weight gradients into main_grad (Megatron-LM), by default False
        """
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if dtype is None:
            dtype = torch.bfloat16

        self.in_features = in_features
        self.out_features = out_features
        self.fp8_dtype = fp8_dtype
        self.use_deepgemm = use_deepgemm
        self.device = device
        self.dtype = dtype
        self.tensor_parallel_mode = tensor_parallel_mode
        self.tensor_parallel_group = tensor_parallel_group
        self.sequence_parallel = sequence_parallel
        self.accumulate_into_main_grad = accumulate_into_main_grad

        # Handle tensor parallelism dimensions
        self.tensor_parallel_size = 1
        self.local_in_features = in_features
        self.local_out_features = out_features

        if tensor_parallel_mode is not None:
            if torch.distributed.is_initialized():
                self.tensor_parallel_size = torch.distributed.get_world_size(tensor_parallel_group)
            else:
                warnings.warn("Distributed not initialized, using tensor_parallel_size=1")
                self.tensor_parallel_size = 1

            if tensor_parallel_mode == "column":
                # Distribute output features across TP ranks
                if out_features % self.tensor_parallel_size != 0:
                    raise ValueError(
                        f"out_features ({out_features}) must be divisible by tensor_parallel_size ({self.tensor_parallel_size})"
                    )
                self.local_out_features = out_features // self.tensor_parallel_size
            elif tensor_parallel_mode == "row":
                # Distribute input features across TP ranks
                if in_features % self.tensor_parallel_size != 0:
                    raise ValueError(
                        f"in_features ({in_features}) must be divisible by tensor_parallel_size ({self.tensor_parallel_size})"
                    )
                self.local_in_features = in_features // self.tensor_parallel_size

        # Create quantizers for input and weight
        self.input_quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=False,  # Will be set to True for wgrad when needed
            block_scaling_dim=block_scaling_dim,
            use_deepgemm_layout=use_deepgemm,
        )

        self.weight_quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=block_scaling_dim,
            use_deepgemm_layout=use_deepgemm,
        )

        # Create weight parameter
        self.weight = nn.Parameter(
            torch.empty(self.local_out_features, self.local_in_features, device=device, dtype=dtype)
        )

        # Create bias if needed
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.local_out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        self.reset_parameters(rng_state_tracker_function)

    def reset_parameters(self, rng_state_tracker_function: Optional[CudaRNGStatesTracker] = None) -> None:
        """Initialize parameters"""
        import math
        import contextlib

        # Initialize values
        init_context = contextlib.nullcontext()
        if rng_state_tracker_function is not None:
            init_context = rng_state_tracker_function().fork()

        with init_context:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass using DeepGEMM-optimized operations.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (..., in_features)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., out_features)
        """
        # Ensure input is on correct device and dtype
        if input.device != self.device:
            input = input.to(self.device)

        # Handle tensor parallelism for input
        if self.tensor_parallel_mode == "column" and self.sequence_parallel:
            # All-gather input across tensor parallel group
            from ..distributed import gather_along_first_dim
            input, _ = gather_along_first_dim(input, self.tensor_parallel_group)

        # Use custom autograd function
        output = _LinearDeepGemmFunction.apply(
            input,
            self.weight,
            self.bias,
            self.input_quantizer,
            self.weight_quantizer,
            self.use_deepgemm,
            self.dtype,
            self.accumulate_into_main_grad,
        )

        # Handle tensor parallelism for output
        if self.tensor_parallel_mode == "row":
            # All-reduce or reduce-scatter output across tensor parallel group
            if self.sequence_parallel:
                from ..distributed import reduce_scatter_along_first_dim
                output, _ = reduce_scatter_along_first_dim(output, self.tensor_parallel_group)
            else:
                torch.distributed.all_reduce(output, group=self.tensor_parallel_group)

        return output

    def extra_repr(self) -> str:
        """Extra representation for printing"""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, fp8_dtype={self.fp8_dtype}, '
                f'use_deepgemm={self.use_deepgemm}, '
                f'tensor_parallel_mode={self.tensor_parallel_mode}, '
                f'accumulate_into_main_grad={self.accumulate_into_main_grad}')


# For backward compatibility, keep the old MoELinearDeepGemm class
class MoELinearDeepGemm(LinearDeepGemm):
    """MoE Linear layer using FP8DeepGemmQuantizer with grouped GEMM support.

    NOTE: This is a placeholder implementation. For full MoE support with
    DeepGEMM grouped GEMM, additional implementation is needed.
    """

    def __init__(self, num_experts: int, **kwargs):
        """Initialize MoELinearDeepGemm module."""
        super().__init__(**kwargs)
        self.num_experts = num_experts
        warnings.warn(
            "MoELinearDeepGemm is not fully implemented. "
            "Use regular LinearDeepGemm for now.",
            UserWarning
        )

    def forward(self, input: torch.Tensor, expert_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass - falls back to regular linear for now."""
        if expert_indices is not None:
            warnings.warn(
                "Expert routing not implemented. Using regular linear forward.",
                UserWarning
            )
        return super().forward(input)