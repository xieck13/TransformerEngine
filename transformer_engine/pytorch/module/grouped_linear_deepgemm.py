# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GroupedLinear module using native DeepGEMM operations for optimized FP8 grouped GEMM."""

from typing import Union, Optional, Callable, Tuple, List
import warnings
import torch
import torch.nn as nn

import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from transformer_engine.common.recipe import Recipe
from .base import (
    get_multi_stream_cublas_workspace,
    TransformerEngineBaseModule,
    get_workspace,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ._common import WeightGradStore, get_module_quantizers
from ..tensor.float8_deepgemm_tensor import FP8DeepGemmQuantizer, FP8DeepGemmQTensor, DEEPGEMM_AVAILABLE
from ..fp8 import FP8GlobalStateManager
from ..utils import (
    divide,
    cast_if_needed,
    clear_tensor_data,
    init_method_constant,
    requires_grad,
    get_default_init_method,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
)
from ..constants import GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..cpu_offload import is_cpu_offload_enabled

if DEEPGEMM_AVAILABLE:
    import deep_gemm

__all__ = ["GroupedLinearDeepGemm"]


class _GroupedLinearDeepGemm(torch.autograd.Function):
    """GroupedLinear with native DeepGEMM operations

    Replaces general_grouped_gemm with DeepGEMM operations for both forward and backward passes,
    implementing 1D1D wgrad with fp32 accumulation for precision.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        use_bias: bool,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        wgrad_store: WeightGradStore,
        input_quantizers: List[FP8DeepGemmQuantizer],
        weight_quantizers: List[FP8DeepGemmQuantizer],
        output_quantizers: List[Optional[FP8DeepGemmQuantizer]],
        grad_output_quantizers: List[Optional[FP8DeepGemmQuantizer]],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        sequence_parallel: bool,
        activation_dtype: torch.dtype,
        is_grad_enabled: bool,
        use_deepgemm: bool,
        accumulate_into_main_grad: bool,
        module,
        *weights_and_biases,
    ) -> torch.Tensor:

        num_gemms = len(m_splits)
        weights = weights_and_biases[:num_gemms]
        biases = weights_and_biases[num_gemms:]
        device = inp.device

        # Check DeepGEMM requirements - raise error if not met
        if not use_deepgemm:
            raise RuntimeError("GroupedLinearDeepGemm requires DeepGEMM to be enabled")

        if not DEEPGEMM_AVAILABLE:
            raise RuntimeError("DeepGEMM is not available but GroupedLinearDeepGemm requires it")

        # Check dimension constraints for all operations
        inp_view = inp.reshape(-1, inp.shape[-1])
        total_batch = inp_view.shape[0]
        in_features = inp_view.shape[1]

        # Verify all tensor dimensions are compatible with DeepGEMM
        if not (total_batch % 128 == 0 and in_features % 128 == 0):
            raise RuntimeError(f"DeepGEMM requirements not met. "
                             f"inp_view: {inp_view.shape} (both dims must be % 128 == 0), "
                             f"All tensor dimensions must be divisible by 128 for DeepGEMM operations.")

        for i, (weight, m_split) in enumerate(zip(weights, m_splits)):
            if not (weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0 and m_split % 128 == 0):
                raise RuntimeError(f"DeepGEMM requirements not met for GEMM {i}. "
                                 f"weight: {weight.shape} (both dims must be % 128 == 0), "
                                 f"m_split: {m_split} (must be % 128 == 0). "
                                 f"All tensor dimensions must be divisible by 128 for DeepGEMM operations.")

        # Store for backward pass
        ctx.save_for_backward(inp, *weights, *biases)
        ctx.m_splits = m_splits
        ctx.num_gemms = num_gemms
        ctx.use_bias = use_bias
        ctx.use_deepgemm = use_deepgemm
        ctx.activation_dtype = activation_dtype
        ctx.is_grad_enabled = is_grad_enabled
        ctx.device = device
        ctx.input_quantizers = input_quantizers
        ctx.weight_quantizers = weight_quantizers
        ctx.accumulate_into_main_grad = accumulate_into_main_grad
        ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation

        # Split input according to m_splits
        input_parts = torch.split(inp_view, m_splits)
        outputs = []

        # Process each GEMM individually using DeepGEMM
        for i, (input_part, weight) in enumerate(zip(input_parts, weights)):
            # Quantize input part
            if input_quantizers[i] is None or not isinstance(input_quantizers[i], FP8DeepGemmQuantizer):
                raise RuntimeError(f"GroupedLinearDeepGemm requires FP8DeepGemmQuantizer for input {i}")

            quantized_input = input_quantizers[i].make_empty(
                input_part.shape, dtype=input_part.dtype, device=device
            )
            input_quantizers[i].update_quantized(input_part, quantized_input)

            # Quantize weight
            if weight_quantizers[i] is None or not isinstance(weight_quantizers[i], FP8DeepGemmQuantizer):
                raise RuntimeError(f"GroupedLinearDeepGemm requires FP8DeepGemmQuantizer for weight {i}")

            quantized_weight = weight_quantizers[i].make_empty(
                weight.shape, dtype=weight.dtype, device=device
            )
            weight_quantizers[i].update_quantized(weight, quantized_weight)

            # DeepGEMM forward pass
            workspace = get_workspace()

            if not (isinstance(quantized_input, FP8DeepGemmQTensor) and
                    isinstance(quantized_weight, FP8DeepGemmQTensor)):
                raise RuntimeError(f"Expected FP8DeepGemmQTensor objects, got {type(quantized_input)}, {type(quantized_weight)}")

            # Create output tensor
            output = torch.empty(
                input_part.shape[0], weight.shape[0],
                dtype=activation_dtype, device=device
            )

            # DeepGEMM NT layout forward: input @ weight.T
            deep_gemm.fp8_gemm_nt(
                (quantized_input.rowwise_data, quantized_input.rowwise_scale_inv),
                (quantized_weight.columnwise_data, quantized_weight.columnwise_scale_inv),
                output,
                c=None,
                recipe=None
            )

            # Add bias if needed
            if use_bias and i < len(biases) and biases[i] is not None:
                output = output + biases[i]

            outputs.append(output)
            print(f"DEBUG: GroupedLinear GEMM {i} forward - input: {input_part.shape}, weight: {weight.shape}, output: {output.shape}")

        # Concatenate all outputs
        out = torch.cat(outputs, dim=0)
        print(f"DEBUG: Successfully used DeepGEMM for GroupedLinear forward with {num_gemms} GEMMs")

        # Return in original input shape format
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using native DeepGEMM operations for both dgrad and wgrad"""
        inp, *weights_and_biases = ctx.saved_tensors
        num_gemms = ctx.num_gemms
        weights = weights_and_biases[:num_gemms]
        biases = weights_and_biases[num_gemms:]

        # Reshape grad_output and input for processing
        grad_output_view = grad_output.contiguous().view(-1, grad_output.shape[-1])
        inp_view = inp.reshape(-1, inp.shape[-1])

        # Split according to m_splits
        grad_outputs = torch.split(grad_output_view, ctx.m_splits)
        input_parts = torch.split(inp_view, ctx.m_splits)

        grad_input_parts = []
        grad_weights = []
        grad_biases = []

        # ==========================================
        # Process each GEMM individually
        # ==========================================
        for i, (grad_out, input_part, weight) in enumerate(zip(grad_outputs, input_parts, weights)):

            # ==========================================
            # Compute dgrad using DeepGEMM
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
                    grad_out.shape, dtype=grad_out.dtype, device=grad_out.device
                )
                grad_output_quantizer.update_quantized(grad_out, grad_output_fp8)

                # Convert weight to FP8 for dgrad - transpose weight for NT layout
                weight_quantizer = FP8DeepGemmQuantizer(
                    TE_DType.kFloat8E4M3,
                    rowwise=True,
                    columnwise=True,
                    use_deepgemm_layout=True,
                )
                # Transpose weight for NT layout
                weight_transposed = weight.t().contiguous()
                weight_fp8 = weight_quantizer.make_empty(
                    weight_transposed.shape, dtype=weight_transposed.dtype, device=weight_transposed.device
                )
                weight_quantizer.update_quantized(weight_transposed, weight_fp8)

                # Create output tensor for dgrad
                grad_input_part = torch.empty(
                    input_part.shape, dtype=ctx.activation_dtype, device=input_part.device
                )

                # dgrad: grad_input = grad_output @ weight (using NT layout)
                deep_gemm.fp8_gemm_nt(
                    (grad_output_fp8.rowwise_data, grad_output_fp8.rowwise_scale_inv),
                    (weight_fp8.columnwise_data, weight_fp8.columnwise_scale_inv),
                    grad_input_part,
                    c=None,
                    recipe=None
                )

                grad_input_parts.append(grad_input_part)
                print(f"DEBUG: GroupedLinear GEMM {i} dgrad - grad_output: {grad_out.shape}, weight_T: {weight_transposed.shape}")

            # ==========================================
            # Compute wgrad using DeepGEMM 1D1D with fp32 accumulation
            # ==========================================
            if len(ctx.needs_input_grad) > (num_gemms + 18) and ctx.needs_input_grad[num_gemms + 18 + i]:  # weight requires grad
                # Determine if we should accumulate into main_grad
                main_grad = None
                if ctx.accumulate_into_main_grad and hasattr(weight, 'main_grad') and weight.main_grad is not None:
                    main_grad = weight.main_grad.detach()

                # Convert input to FP8 for wgrad - transpose for NT layout
                input_transposed = input_part.t().contiguous()
                input_quantizer = FP8DeepGemmQuantizer(
                    TE_DType.kFloat8E4M3,
                    rowwise=True,
                    columnwise=True,  # Use columnwise for B tensor in NT layout
                    use_deepgemm_layout=True,
                )
                input_fp8 = input_quantizer.make_empty(
                    input_transposed.shape, dtype=input_transposed.dtype, device=input_transposed.device
                )
                input_quantizer.update_quantized(input_transposed, input_fp8)

                # Convert grad_output to FP8 for wgrad - transpose for NT layout
                grad_output_quantizer_wgrad = FP8DeepGemmQuantizer(
                    TE_DType.kFloat8E4M3,
                    rowwise=True,
                    columnwise=False,  # Use rowwise for A tensor in NT layout
                    use_deepgemm_layout=True,
                )
                grad_output_transposed = grad_out.t().contiguous()
                grad_output_fp8_wgrad = grad_output_quantizer_wgrad.make_empty(
                    grad_output_transposed.shape, dtype=grad_output_transposed.dtype, device=grad_output_transposed.device
                )
                grad_output_quantizer_wgrad.update_quantized(grad_output_transposed, grad_output_fp8_wgrad)

                # Create output tensor for wgrad
                use_accumulation = main_grad is not None
                if use_accumulation:
                    # For main_grad accumulation, use bfloat16 output and handle fp32 accumulation in software
                    grad_weight_out = torch.empty(
                        weight.shape, dtype=ctx.activation_dtype, device=weight.device
                    )
                else:
                    grad_weight_out = torch.empty(
                        weight.shape, dtype=ctx.activation_dtype, device=weight.device
                    )

                # wgrad: grad_weight = grad_output.T @ input (NT layout)
                print(f"DEBUG: GroupedLinear GEMM {i} wgrad 1D1D - grad_output: {grad_out.shape} -> transposed: {grad_output_transposed.shape}, input: {input_part.shape} -> transposed: {input_transposed.shape}, expected output: {weight.shape}")

                deep_gemm.fp8_gemm_nt(
                    (grad_output_fp8_wgrad.rowwise_data, grad_output_fp8_wgrad.rowwise_scale_inv),  # A tensor
                    (input_fp8.columnwise_data, input_fp8.columnwise_scale_inv),  # B tensor
                    grad_weight_out,
                    c=None,  # Start with basic case, no accumulation
                    recipe=None  # Start with basic case, no forced recipe
                )

                if use_accumulation:
                    # Accumulate into main_grad in fp32 for precision
                    main_grad.add_(grad_weight_out.to(torch.float32))
                    print(f"DEBUG: GroupedLinear GEMM {i} wgrad 1D1D with fp32 accumulation")
                    grad_weights.append(None)  # Don't return grad_weight when accumulating into main_grad
                else:
                    print(f"DEBUG: GroupedLinear GEMM {i} wgrad 1D1D")
                    grad_weights.append(grad_weight_out)
            else:
                grad_weights.append(None)

            # ==========================================
            # Compute grad bias
            # ==========================================
            if ctx.use_bias and len(biases) > i and biases[i] is not None:
                grad_bias = grad_out.sum(dim=0)
                grad_biases.append(grad_bias)
            else:
                grad_biases.append(None)

        # Concatenate input gradients if needed
        if ctx.needs_input_grad[0]:
            grad_input = torch.cat(grad_input_parts, dim=0).view(inp.shape)
        else:
            grad_input = None

        print(f"DEBUG: Successfully used DeepGEMM for GroupedLinear backward with {num_gemms} GEMMs")

        return (grad_input, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, *grad_weights, *grad_biases)


class GroupedLinearDeepGemm(TransformerEngineBaseModule):
    """GroupedLinear layer using native DeepGEMM operations for optimized FP8 grouped GEMM.

    This module replaces general_grouped_gemm with native DeepGEMM operations,
    implementing 1D1D wgrad with fp32 accumulation for precision enhancement.

    Key features:
    - Native DeepGEMM operations for forward and backward passes
    - 1D1D kernel preference for weight gradient computation with fp32 accumulation
    - Megatron-LM main_grad compatibility
    - Strict DeepGEMM dimension constraint enforcement
    - No fallback logic - raises errors when constraints aren't met

    Example usage:
    ```python
    # Create a grouped linear layer with DeepGEMM optimization
    grouped_linear = GroupedLinearDeepGemm(
        num_gemms=8,
        in_features=4096,
        out_features=4096,
        fp8_dtype=TE_DType.kFloat8E4M3,
        use_bias=True,
        accumulate_into_main_grad=False  # Set True for Megatron-LM
    )

    # Forward pass
    input_tensor = torch.randn(128, 4096, device='cuda', dtype=torch.bfloat16)
    m_splits = [128, 128, 128, 128, 128, 128, 128, 128]  # 8 experts, 128 tokens each
    output = grouped_linear(input_tensor, m_splits)
    ```
    """

    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        rng_tracker_name: Optional[str] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        device: Union[torch.device, str] = "cuda",
        fp8_dtype: TE_DType = TE_DType.kFloat8E4M3,
        block_scaling_dim: int = 2,
        delay_wgrad_compute: bool = False,
        save_original_input: bool = False,
        accumulate_into_main_grad: bool = False,
        **kwargs
    ) -> None:
        """Initialize GroupedLinearDeepGemm module.

        Parameters
        ----------
        num_gemms : int
            Number of GEMMs to be performed simultaneously
        in_features : int
            Size of each input sample
        out_features : int
            Size of each output sample
        accumulate_into_main_grad : bool, optional
            Whether to accumulate weight gradients into main_grad (Megatron-LM), by default False
        [... other parameters same as base GroupedLinear ...]
        """
        super().__init__()

        if not DEEPGEMM_AVAILABLE:
            raise RuntimeError("DeepGEMM is required for GroupedLinearDeepGemm but is not available")

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.save_original_input = save_original_input
        self.fp8_dtype = fp8_dtype
        self.block_scaling_dim = block_scaling_dim
        self.accumulate_into_main_grad = accumulate_into_main_grad

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        if params_dtype is None:
            params_dtype = torch.bfloat16

        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name

        self.wgrad_store = WeightGradStore(delay_wgrad_compute)

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)

        if self.tp_size > 1 and bias:
            raise ValueError(
                "GroupedLinearDeepGemm doesn't support bias when TP > 1. "
                "Because the TP communication is handled outside of this module."
            )

        self.parallel_mode = parallel_mode
        assert (
            self.parallel_mode in GemmParallelModes
        ), f"parallel_mode {parallel_mode} not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        # Create quantizers for DeepGEMM - all required for native operations
        self.input_quantizers = []
        self.weight_quantizers = []
        for i in range(self.num_gemms):
            # Input quantizer
            input_quantizer = FP8DeepGemmQuantizer(
                fp8_dtype=fp8_dtype,
                rowwise=True,
                columnwise=False,
                block_scaling_dim=block_scaling_dim,
                use_deepgemm_layout=True,
            )
            self.input_quantizers.append(input_quantizer)

            # Weight quantizer
            weight_quantizer = FP8DeepGemmQuantizer(
                fp8_dtype=fp8_dtype,
                rowwise=True,
                columnwise=True,
                block_scaling_dim=block_scaling_dim,
                use_deepgemm_layout=True,
            )
            self.weight_quantizers.append(weight_quantizer)

        # Additional quantizers (can be set by recipes)
        self.output_quantizers = [None] * self.num_gemms
        self.grad_output_quantizers = [None] * self.num_gemms

        # Create weight and bias parameters
        for i in range(self.num_gemms):
            # Weight parameter
            weight = nn.Parameter(
                torch.empty(
                    self.out_features,
                    self.in_features,
                    device=self.device,
                    dtype=params_dtype,
                )
            )
            setattr(self, f"weight{i}", weight)

            # Bias parameter
            if self.use_bias:
                bias_param = nn.Parameter(
                    torch.empty(
                        self.out_features,
                        device=self.device,
                        dtype=params_dtype,
                    )
                )
                setattr(self, f"bias{i}", bias_param)
            else:
                bias_tensor = torch.tensor([], dtype=params_dtype, device=self.device)
                setattr(self, f"bias{i}", bias_tensor)

        # Initialize parameters
        if init_method is None:
            init_method = get_default_init_method()

        for i in range(self.num_gemms):
            init_method(getattr(self, f"weight{i}"))
            if self.use_bias:
                nn.init.zeros_(getattr(self, f"bias{i}"))

        # Set tensor parallel attributes
        for i in range(self.num_gemms):
            set_tensor_model_parallel_attributes(
                tensor=getattr(self, f"weight{i}"),
                is_parallel=True,
                dim=1 if self.parallel_mode == "row" else 0,
                stride=1,
            )

            if self.use_bias:
                if self.parallel_mode == "row":
                    setattr(
                        getattr(self, f"bias{i}"),
                        "sequence_parallel",
                        self.sequence_parallel,
                    )
                elif self.parallel_mode == "column":
                    set_tensor_model_parallel_attributes(getattr(self, f"bias{i}"), True, 0, 1)

        # Other configuration
        self.activation_dtype = params_dtype
        self.cpu_offloading = False

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass using native DeepGEMM operations.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor
        m_splits : List[int]
            List of integers representing the split of the input tensor
        is_first_microbatch : Optional[bool], optional
            Whether this is the first microbatch, by default None

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            Output tensor(s) - single tensor or tuple if return_bias=True
        """
        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."

        # Ensure input is contiguous and on correct device/dtype
        if not inp.is_contiguous():
            inp = inp.contiguous()
        if inp.device != self.device:
            inp = inp.to(self.device)

        # Determine if we're in FP8 mode
        fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
        fp8_calibration = FP8GlobalStateManager.is_fp8_calibration()

        # Get weight and bias tensors
        weight_tensors = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        bias_tensors = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]

        out = _GroupedLinearDeepGemm.apply(
            inp,
            m_splits,
            self.apply_bias,
            is_first_microbatch,
            fp8_enabled,
            fp8_calibration,
            self.wgrad_store,
            self.input_quantizers,
            self.weight_quantizers,
            self.output_quantizers,
            self.grad_output_quantizers,
            self.fuse_wgrad_accumulation,
            self.cpu_offloading,
            self.sequence_parallel,
            self.activation_dtype,
            torch.is_grad_enabled(),
            True,  # use_deepgemm - always True for GroupedLinearDeepGemm
            self.accumulate_into_main_grad,
            self,
            *weight_tensors,
            *bias_tensors,
        )

        if self.return_bias:
            return out, [cast_if_needed(b, self.activation_dtype) for b in bias_tensors]
        return out

    def set_tensor_parallel_group(self, tp_group: dist_group_type) -> None:
        """Set tensor parallel group"""
        pass  # Placeholder implementation

    def extra_repr(self) -> str:
        """Extra representation for debugging"""
        return (f'num_gemms={self.num_gemms}, in_features={self.in_features}, '
                f'out_features={self.out_features}, bias={self.use_bias}, '
                f'fp8_dtype={self.fp8_dtype}, '
                f'accumulate_into_main_grad={self.accumulate_into_main_grad}')