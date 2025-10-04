# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GroupedLinear module using FP8DeepGemmQuantizer for optimized FP8 operations."""

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
from ..cpp_extensions.deepgemm import deepgemm_fp8_grouped_gemm
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

__all__ = ["GroupedLinearDeepGemm"]


class _GroupedLinearDeepGemm(torch.autograd.Function):
    """GroupedLinear with DeepGEMM optimization

    This function extends the regular GroupedLinear with DeepGEMM-optimized
    grouped matrix multiplication operations.
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
        module,
        *weights_and_biases,
    ) -> torch.Tensor:

        num_gemms = len(m_splits)
        weights = weights_and_biases[:num_gemms]
        biases = weights_and_biases[num_gemms:]
        device = inp.device

        # Use regular GroupedLinear if DeepGEMM not available or disabled
        if not use_deepgemm or not DEEPGEMM_AVAILABLE:
            from .grouped_linear import _GroupedLinear
            return _GroupedLinear.forward(
                ctx, inp, m_splits, use_bias, is_first_microbatch, fp8, fp8_calibration,
                wgrad_store, input_quantizers, weight_quantizers, output_quantizers,
                grad_output_quantizers, fuse_wgrad_accumulation, cpu_offloading,
                sequence_parallel, activation_dtype, is_grad_enabled, module, None,
                False, *weights_and_biases
            )

        # Store for backward pass
        ctx.save_for_backward(inp, *weights, *biases)
        ctx.m_splits = m_splits
        ctx.num_gemms = num_gemms
        ctx.use_bias = use_bias
        ctx.use_deepgemm = use_deepgemm
        ctx.activation_dtype = activation_dtype
        ctx.is_grad_enabled = is_grad_enabled
        ctx.device = device

        # Initialize input tensors
        in_features = weights[0].size(-1)
        if inp.size(-1) != in_features:
            raise ValueError(
                f"Input tensor (shape={tuple(inp.size())}) is not compatible with "
                f"weight tensor (shape={tuple(weights[0].size())})"
            )
        inp_view = inp.reshape(-1, in_features)

        # Quantize inputs using DeepGEMM quantizers
        quantized_inputs = []
        for i, m_split in enumerate(m_splits):
            # Split input for this GEMM
            start_idx = sum(m_splits[:i])
            end_idx = start_idx + m_split
            input_slice = inp_view[start_idx:end_idx]

            if input_quantizers[i] is not None:
                # Create quantized input tensor
                quantized_input = input_quantizers[i].make_empty(
                    input_slice.shape,
                    dtype=input_slice.dtype,
                    device=device
                )
                input_quantizers[i].update_quantized(input_slice, quantized_input)
                quantized_inputs.append(quantized_input)
            else:
                quantized_inputs.append(input_slice)

        # Quantize weights using DeepGEMM quantizers
        quantized_weights = []
        for i, weight in enumerate(weights):
            if weight_quantizers[i] is not None:
                # Create quantized weight tensor
                quantized_weight = weight_quantizers[i].make_empty(
                    weight.shape,
                    dtype=weight.dtype,
                    device=device
                )
                weight_quantizers[i].update_quantized(weight, quantized_weight)
                quantized_weights.append(quantized_weight)
            else:
                quantized_weights.append(weight)

        # Get workspace tensor
        workspace = get_workspace()

        # Initialize output tensor
        out = torch.empty(
            [sum(m_splits), weights[0].size(0)],
            dtype=activation_dtype,
            device=device,
        )

        # Check if we can use DeepGEMM grouped GEMM
        use_deepgemm_grouped = (
            all(isinstance(qi, FP8DeepGemmQTensor) for qi in quantized_inputs) and
            all(isinstance(qw, FP8DeepGemmQTensor) for qw in quantized_weights)
        )

        if use_deepgemm_grouped:
            try:
                # Prepare inputs for grouped GEMM
                # Concatenate all quantized inputs
                all_inputs = torch.cat(quantized_inputs, dim=0)
                # Stack all quantized weights
                all_weights = torch.stack(quantized_weights, dim=0)

                # Convert m_splits to tensor
                m_splits_tensor = torch.tensor(m_splits, device=device, dtype=torch.long)

                # Prepare bias
                bias_tensor = None
                if use_bias and biases[0] is not None:
                    bias_tensor = torch.stack(biases, dim=0)

                # Call DeepGEMM grouped GEMM
                output, _ = deepgemm_fp8_grouped_gemm(
                    all_inputs,
                    all_weights,
                    workspace,
                    m_splits_tensor,
                    layout="nt",
                    bias=bias_tensor,
                    out_dtype=activation_dtype
                )
                out = output

            except Exception as e:
                warnings.warn(f"DeepGEMM grouped GEMM failed, falling back to individual GEMMs: {e}")
                # Fall back to individual GEMMs
                use_deepgemm_grouped = False

        if not use_deepgemm_grouped:
            # Perform individual GEMMs
            outputs = []
            for i, (quantized_input, quantized_weight) in enumerate(zip(quantized_inputs, quantized_weights)):
                # Dequantize if needed
                if isinstance(quantized_input, FP8DeepGemmQTensor):
                    input_data = quantized_input.dequantize()
                else:
                    input_data = quantized_input

                if isinstance(quantized_weight, FP8DeepGemmQTensor):
                    weight_data = quantized_weight.dequantize()
                else:
                    weight_data = quantized_weight

                # Perform regular GEMM
                gemm_output = torch.matmul(input_data, weight_data.T)

                # Add bias if needed
                if use_bias and biases[i] is not None:
                    gemm_output = gemm_output + biases[i]

                outputs.append(gemm_output)

            # Concatenate outputs
            out = torch.cat(outputs, dim=0)

        # Return in original input shape format
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        # For now, use standard backward pass
        # In a full implementation, this would be optimized for DeepGEMM grouped operations
        inp, *weights_and_biases = ctx.saved_tensors
        num_gemms = ctx.num_gemms
        weights = weights_and_biases[:num_gemms]
        biases = weights_and_biases[num_gemms:]

        # Compute gradients using standard autograd
        # This is a simplified implementation - production would need more optimization
        grad_output_view = grad_output.contiguous().view(-1, grad_output.shape[-1])

        # Split grad_output according to m_splits
        grad_outputs = torch.split(grad_output_view, ctx.m_splits)

        # Compute gradients
        grad_inp_parts = []
        grad_weights = []
        grad_biases = []

        inp_view = inp.reshape(-1, inp.shape[-1])
        inp_parts = torch.split(inp_view, ctx.m_splits)

        for i, (grad_out, weight, inp_part) in enumerate(zip(grad_outputs, weights, inp_parts)):
            # Weight gradient
            grad_weight = torch.matmul(grad_out.transpose(-2, -1), inp_part)
            grad_weights.append(grad_weight)

            # Input gradient
            grad_inp_part = torch.matmul(grad_out, weight)
            grad_inp_parts.append(grad_inp_part)

            # Bias gradient
            if ctx.use_bias and len(biases) > i and biases[i] is not None:
                grad_bias = grad_out.sum(dim=0)
                grad_biases.append(grad_bias)
            else:
                grad_biases.append(None)

        # Concatenate input gradients
        grad_inp = torch.cat(grad_inp_parts, dim=0).view(inp.shape)

        return (grad_inp, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, *grad_weights, *grad_biases)


class GroupedLinearDeepGemm(TransformerEngineBaseModule):
    """GroupedLinear layer using FP8DeepGemmQuantizer for optimized FP8 operations.

    This module extends the regular GroupedLinear with DeepGEMM optimization,
    using FP8DeepGemmQuantizer's optimized kernels for grouped matrix operations.

    Example usage:
    ```python
    # Create a grouped linear layer with DeepGEMM optimization
    grouped_linear = GroupedLinearDeepGemm(
        num_gemms=8,
        in_features=4096,
        out_features=4096,
        fp8_dtype=TE_DType.kFloat8E4M3,
        use_bias=True,
        use_deepgemm=True
    )

    # Forward pass
    input_tensor = torch.randn(128, 4096, device='cuda', dtype=torch.bfloat16)
    m_splits = [16, 16, 16, 16, 16, 16, 16, 16]  # 8 experts, 16 tokens each
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
        use_deepgemm: bool = True,
        fp8_dtype: TE_DType = TE_DType.kFloat8E4M3,
        block_scaling_dim: int = 2,
        delay_wgrad_compute: bool = False,
        save_original_input: bool = False,
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
        sequence_parallel : bool, optional
            Whether to use sequence parallelism, by default False
        fuse_wgrad_accumulation : bool, optional
            Whether to fuse weight gradient accumulation, by default False
        tp_group : Optional[dist_group_type], optional
            Tensor parallel group, by default None
        tp_size : int, optional
            Tensor parallel size, by default 1
        get_rng_state_tracker : Optional[Callable], optional
            RNG state tracker, by default None
        rng_tracker_name : Optional[str], optional
            RNG tracker name, by default None
        init_method : Optional[Callable], optional
            Weight initialization method, by default None
        bias : bool, optional
            Whether to use bias, by default True
        return_bias : bool, optional
            Whether to return bias, by default False
        params_dtype : Optional[torch.dtype], optional
            Parameter dtype, by default None
        parallel_mode : Optional[str], optional
            Parallel mode, by default None
        device : Union[torch.device, str], optional
            Device, by default "cuda"
        use_deepgemm : bool, optional
            Whether to use DeepGEMM optimization, by default True
        fp8_dtype : TE_DType, optional
            FP8 data type, by default TE_DType.kFloat8E4M3
        block_scaling_dim : int, optional
            Block scaling dimension, by default 2
        delay_wgrad_compute : bool, optional
            Whether to delay weight gradient computation, by default False
        save_original_input : bool, optional
            Whether to save original input, by default False
        """
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.save_original_input = save_original_input
        self.use_deepgemm = use_deepgemm and DEEPGEMM_AVAILABLE
        self.fp8_dtype = fp8_dtype
        self.block_scaling_dim = block_scaling_dim

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

        # Create quantizers for DeepGEMM
        if self.use_deepgemm:
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
        else:
            self.input_quantizers = [None] * self.num_gemms
            self.weight_quantizers = [None] * self.num_gemms

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
        """Forward pass using DeepGEMM-optimized grouped operations.

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
            self.use_deepgemm,
            self,
            *weight_tensors,
            *bias_tensors,
        )

        if self.return_bias:
            return out, [cast_if_needed(b, self.activation_dtype) for b in bias_tensors]
        return out

    def backward_dw(self):
        """Execute delayed weight gradient computation."""
        if self.wgrad_store is None or not self.wgrad_store.delay_wgrad_compute():
            return
        with torch.cuda.nvtx.range("_GroupedLinearDeepGemm_wgrad"):
            (_, grad_biases_, _), tensor_list = self.wgrad_store.pop()
            wgrad_list = tensor_list[2]
            weight_params = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
            bias_params = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]

            if not self.fuse_wgrad_accumulation:
                for i in range(self.num_gemms):
                    weight_params[i].grad = wgrad_list[i].to(weight_params[i].dtype)

            if self.use_bias:
                for i in range(self.num_gemms):
                    if bias_params[i].grad is None:
                        bias_params[i].grad = grad_biases_[i].to(bias_params[i].dtype)

            del grad_biases_
            del wgrad_list
            del tensor_list

    def extra_repr(self) -> str:
        """Extra representation for debugging"""
        return (f'num_gemms={self.num_gemms}, in_features={self.in_features}, '
                f'out_features={self.out_features}, bias={self.use_bias}, '
                f'use_deepgemm={self.use_deepgemm}, fp8_dtype={self.fp8_dtype}')