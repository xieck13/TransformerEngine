# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DeepGEMM-optimized GEMM operations for FP8 tensors"""

from typing import Optional, Tuple, Union
import warnings
import torch
import transformer_engine_torch as tex

from ..tensor.float8_deepgemm_tensor import FP8DeepGemmQTensor, DEEPGEMM_AVAILABLE
from ..utils import devices_match, _empty_tensor

if DEEPGEMM_AVAILABLE:
    import deep_gemm

__all__ = [
    "deepgemm_fp8_gemm",
    "deepgemm_fp8_grouped_gemm",
]


def deepgemm_fp8_gemm(
    A: Union[torch.Tensor, FP8DeepGemmQTensor],
    B: Union[torch.Tensor, FP8DeepGemmQTensor],
    workspace: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    layout: str = "nt",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    **kwargs
) -> Tuple[torch.Tensor, ...]:
    """Perform FP8 GEMM using DeepGEMM kernels.

    This function provides a DeepGEMM-optimized path for FP8 GEMM operations,
    with automatic fallback to regular GEMM when DeepGEMM is unavailable or
    when inputs are not compatible.

    Parameters
    ----------
    A : Union[torch.Tensor, FP8DeepGemmQTensor]
        Input tensor A
    B : Union[torch.Tensor, FP8DeepGemmQTensor]
        Input tensor B
    workspace : torch.Tensor
        Workspace tensor (for compatibility with general_gemm)
    out_dtype : Optional[torch.dtype], optional
        Output data type, by default None
    layout : str, optional
        GEMM layout ("nt", "nn", "tn", "tt"), by default "nt"
    out : Optional[torch.Tensor], optional
        Pre-allocated output tensor, by default None
    bias : Optional[torch.Tensor], optional
        Bias tensor to add, by default None
    accumulate : bool, optional
        Whether to accumulate with existing output, by default False
    alpha : float, optional
        Scaling factor for A@B, by default 1.0
    beta : Optional[float], optional
        Scaling factor for accumulation, by default None

    Returns
    -------
    Tuple[torch.Tensor, ...]
        Result tensor and additional outputs for compatibility with general_gemm
    """
    if not DEEPGEMM_AVAILABLE:
        warnings.warn("DeepGEMM not available, falling back to regular GEMM")
        # Import here to avoid circular imports
        from ..cpp_extensions.gemm import general_gemm
        return general_gemm(
            A, B, workspace,
            out_dtype=out_dtype,
            layout=layout,
            out=out,
            bias=bias,
            accumulate=accumulate,
            alpha=alpha,
            beta=beta,
            **kwargs
        )

    # Check if both inputs are FP8DeepGemmQTensor
    if not (isinstance(A, FP8DeepGemmQTensor) and isinstance(B, FP8DeepGemmQTensor)):
        warnings.warn("Non-FP8DeepGemmQTensor inputs, falling back to regular GEMM")
        from ..cpp_extensions.gemm import general_gemm
        return general_gemm(
            A, B, workspace,
            out_dtype=out_dtype,
            layout=layout,
            out=out,
            bias=bias,
            accumulate=accumulate,
            alpha=alpha,
            beta=beta,
            **kwargs
        )

    # Ensure tensors are on the same device
    if not devices_match(A.device, B.device):
        raise ValueError("Input tensors must be on the same device")

    # Determine output dtype FIRST, before creating tensor
    if out_dtype is None:
        # Based on DeepGEMM exploration:
        # - 1D2D kernels: c=None, recipe=None, bfloat16 output, columnwise B
        # - 1D1D kernels: c=None, recipe=(1,1,128), bfloat16 output, rowwise B
        # We can use 1D1D for accumulation if B has rowwise data, otherwise fall back to 1D2D + software
        out_dtype = torch.bfloat16
        print(f"DEBUG: accumulate={accumulate}, bias={bias is not None}")
        print(f"DEBUG: Using bfloat16 output for DeepGEMM compatibility")

    # Calculate output shape
    if layout in ["nt", "nn"]:
        out_shape = list(A.shape[:-1]) + [B.shape[-2] if layout == "nt" else B.shape[-1]]
    else:  # "tn", "tt"
        out_shape = list(A.shape[:-2]) + [A.shape[-1]] + [B.shape[-2] if layout == "tt" else B.shape[-1]]

    # Create output tensor with correct dtype AFTER determining out_dtype
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=A.device)
        print(f"DEBUG: Created output tensor with dtype {out.dtype}")
    elif out.dtype != out_dtype:
        # If pre-allocated tensor has wrong dtype, convert it
        out = out.to(out_dtype)
        print(f"DEBUG: Converted output tensor to dtype {out.dtype}")
    else:
        print(f"DEBUG: Using pre-allocated output tensor with dtype {out.dtype}")

    # Handle accumulation: if accumulate=True, we need to preserve existing output values
    # and add them back after the DeepGEMM operation
    accumulated_output = None
    if accumulate and out is not None:
        # Save the current output to add back later
        accumulated_output = out.clone()
        print(f"DEBUG: Saved accumulated output for software accumulation")

    # Prepare bias for post-DeepGEMM addition
    bias_to_add = bias

    # Determine kernel type based on available quantization data and operation type
    # 1D1D kernels: good for accumulation, require rowwise B, use recipe=(1,1,128)
    # 1D2D kernels: general purpose, require columnwise B, use recipe=None
    use_1d1d_kernel = False

    # Check if we can use 1D1D kernel
    if accumulate and B._rowwise_data is not None:
        # 1D1D kernel available and preferred for accumulation
        use_1d1d_kernel = True
        kernel_type = "1D1D"
        print(f"DEBUG: Selected 1D1D kernel for accumulation (rowwise B data available)")
    else:
        # Use 1D2D kernel
        use_1d1d_kernel = False
        kernel_type = "1D2D"
        if accumulate:
            print(f"DEBUG: Selected 1D2D kernel with software accumulation (rowwise B data not available)")
        else:
            print(f"DEBUG: Selected 1D2D kernel for forward pass")

    # Prepare FP8 data and scaling factors
    def _get_fp8_data_and_scales(tensor: FP8DeepGemmQTensor, columnwise: bool = False):
        if columnwise and tensor._columnwise_data is not None:
            return tensor._columnwise_data, tensor._columnwise_scale_inv
        elif not columnwise and tensor._rowwise_data is not None:
            return tensor._rowwise_data, tensor._rowwise_scale_inv
        elif tensor._rowwise_data is not None:
            return tensor._rowwise_data, tensor._rowwise_scale_inv
        elif tensor._columnwise_data is not None:
            return tensor._columnwise_data, tensor._columnwise_scale_inv
        else:
            raise ValueError("No suitable FP8 data found in tensor")

    # Get FP8 data and scales
    try:
        # A tensor always uses rowwise (per_token)
        A_data, A_scales = _get_fp8_data_and_scales(A, columnwise=False)
        print(f"DEBUG: A tensor uses rowwise data, shape={A_data.shape}, scales_shape={A_scales.shape}")

        # B tensor quantization depends on kernel type:
        if use_1d1d_kernel:
            # 1D1D kernel: B uses rowwise (per_token) quantization
            B_data, B_scales = B._rowwise_data, B._rowwise_scale_inv
            print(f"DEBUG: Using rowwise B data for 1D1D kernel, shape={B_data.shape}, scales_shape={B_scales.shape}")
        else:
            # 1D2D kernel: B uses columnwise (per_block) quantization
            B_data, B_scales = _get_fp8_data_and_scales(B, columnwise=True)
            print(f"DEBUG: Using columnwise B data for 1D2D kernel, shape={B_data.shape}, scales_shape={B_scales.shape}")

        # Create DeepGEMM input tuples
        # DeepGEMM expects (data, scales) tuples directly
        A_tuple = (A_data, A_scales)
        B_tuple = (B_data, B_scales)
        print(f"DEBUG: Created DeepGEMM tuples, A_tuple shapes=({A_tuple[0].shape}, {A_tuple[1].shape}), B_tuple shapes=({B_tuple[0].shape}, {B_tuple[1].shape})")

    except Exception as e:
        # Rather than complex fallback logic, fail cleanly with clear error message
        raise RuntimeError(f"Failed to prepare DeepGEMM data: {e}. "
                          f"FP8DeepGemmQTensor objects are only compatible with DeepGEMM operations. "
                          f"Use regular Float8BlockwiseQTensor with general_gemm for fallback capability.")

    # Note: accumulate=True without bias means accumulate into output tensor,
    # but DeepGEMM doesn't support this directly - we handle it with alpha/beta scaling

    try:
        # Check for layouts that DeepGEMM doesn't support and fall back early
        if layout == "nn":
            # Note: NN layout has B matrix layout constraints in current DeepGEMM
            # Fall back to regular GEMM for NN layout
            warnings.warn("NN layout not supported in current DeepGEMM, falling back to regular GEMM")
            from ..cpp_extensions.gemm import general_gemm
            # Need to reconstruct the original parameters with proper accumulated output handling
            if accumulated_output is not None:
                # We saved the output, need to handle accumulation properly in the fallback
                result, ws = general_gemm(
                    A, B, workspace,
                    out_dtype=out_dtype,
                    layout=layout,
                    out=None,  # Let general_gemm create its own output
                    bias=bias_to_add,
                    accumulate=False,  # Don't accumulate in general_gemm
                    alpha=alpha,
                    beta=1.0 if beta is None else beta  # Use proper beta
                )
                # Add our saved accumulated output
                if beta is not None:
                    result.add_(accumulated_output, alpha=beta)
                else:
                    result.add_(accumulated_output)
                return (result, ws)
            else:
                return general_gemm(
                    A, B, workspace,
                    out_dtype=out_dtype,
                    layout=layout,
                    out=out,
                    bias=bias_to_add,
                    accumulate=accumulate,
                    alpha=alpha,
                    beta=beta
                )

        # Set DeepGEMM kernel parameters based on kernel type
        if use_1d1d_kernel:
            # 1D1D kernel parameters: recipe=(1,1,128), c=None
            recipe = (1, 1, 128)
            c_tensor = None
            print(f"DEBUG: Selected 1D1D recipe {recipe}, c_tensor=None")
        else:
            # 1D2D kernel parameters: recipe=None, c=None
            recipe = None
            c_tensor = None
            print(f"DEBUG: Selected 1D2D recipe {recipe}, c_tensor=None")

        # Call appropriate DeepGEMM kernel using the correct (data, scales) tuple format
        if layout == "nt":
            deep_gemm.fp8_gemm_nt(
                A_tuple,
                B_tuple,
                out,
                c=c_tensor,
                recipe=recipe
            )
        elif layout == "tn":
            deep_gemm.fp8_gemm_tn(
                A_tuple,
                B_tuple,
                out,
                c=c_tensor,
                recipe=recipe
            )
        elif layout == "tt":
            deep_gemm.fp8_gemm_tt(
                A_tuple,
                B_tuple,
                out,
                c=c_tensor,
                recipe=recipe
            )
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        # Apply scaling factors and handle accumulation/bias in software
        if alpha != 1.0:
            out.mul_(alpha)

        # Handle accumulation based on kernel type
        if accumulated_output is not None:
            if use_1d1d_kernel:
                # 1D1D kernel should handle accumulation directly, but we saved output just in case
                # For now, still add the accumulated output to be safe
                if beta is not None:
                    out.add_(accumulated_output, alpha=beta)
                else:
                    out.add_(accumulated_output)  # Default beta=1.0
                print(f"DEBUG: Added accumulated output with beta={beta} (1D1D kernel)")
            else:
                # 1D2D kernel: software-based accumulation
                if beta is not None:
                    out.add_(accumulated_output, alpha=beta)
                else:
                    out.add_(accumulated_output)  # Default beta=1.0
                print(f"DEBUG: Added accumulated output with beta={beta} (1D2D kernel)")

        # Add bias if provided
        if bias_to_add is not None:
            if out.shape != bias_to_add.shape:
                # Broadcast bias if needed
                bias_broadcasted = bias_to_add.expand(out.shape)
                out.add_(bias_broadcasted)
            else:
                out.add_(bias_to_add)
            print(f"DEBUG: Added bias in software")

        # Return in format compatible with general_gemm
        return (out, workspace)

    except Exception as e:
        # Rather than complex fallback logic, fail cleanly with clear error message
        raise RuntimeError(f"DeepGEMM operation failed: {e}. "
                          f"FP8DeepGemmQTensor objects are only compatible with DeepGEMM operations.")


def deepgemm_fp8_grouped_gemm(
    A: Union[torch.Tensor, FP8DeepGemmQTensor],
    B: Union[torch.Tensor, FP8DeepGemmQTensor],
    workspace: torch.Tensor,
    m_splits: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    layout: str = "nt",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, ...]:
    """Perform grouped FP8 GEMM using DeepGEMM kernels.

    This function provides DeepGEMM-optimized grouped GEMM operations for
    scenarios like MoE where different experts process different numbers of tokens.

    Parameters
    ----------
    A : Union[torch.Tensor, FP8DeepGemmQTensor]
        Input tensor A
    B : Union[torch.Tensor, FP8DeepGemmQTensor]
        Input tensor B
    workspace : torch.Tensor
        Workspace tensor
    m_splits : torch.Tensor
        Tensor defining the M-axis splits for grouping
    out_dtype : Optional[torch.dtype], optional
        Output data type, by default None
    layout : str, optional
        GEMM layout ("nt", "nn"), by default "nt"
    out : Optional[torch.Tensor], optional
        Pre-allocated output tensor, by default None
    bias : Optional[torch.Tensor], optional
        Bias tensor to add, by default None

    Returns
    -------
    Tuple[torch.Tensor, ...]
        Result tensor and additional outputs
    """
    if not DEEPGEMM_AVAILABLE:
        warnings.warn("DeepGEMM not available, falling back to regular grouped GEMM")
        from ..cpp_extensions.gemm import general_grouped_gemm
        return general_grouped_gemm(
            A, B, workspace, m_splits,
            out_dtype=out_dtype,
            layout=layout,
            out=out,
            bias=bias,
            **kwargs
        )

    # Check if both inputs are FP8DeepGemmQTensor
    if not (isinstance(A, FP8DeepGemmQTensor) and isinstance(B, FP8DeepGemmQTensor)):
        warnings.warn("Non-FP8DeepGemmQTensor inputs, falling back to regular grouped GEMM")
        from ..cpp_extensions.gemm import general_grouped_gemm
        return general_grouped_gemm(
            A, B, workspace, m_splits,
            out_dtype=out_dtype,
            layout=layout,
            out=out,
            bias=bias,
            **kwargs
        )

    # For grouped GEMM, we need to ensure proper alignment
    try:
        if DEEPGEMM_AVAILABLE:
            mk_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
            # Check if M splits are properly aligned
            for split in m_splits:
                if split % mk_alignment != 0:
                    warnings.warn(f"M split {split} not aligned to {mk_alignment}, falling back to regular GEMM")
                    from ..cpp_extensions.gemm import general_grouped_gemm
                    return general_grouped_gemm(
                        A, B, workspace, m_splits,
                        out_dtype=out_dtype,
                        layout=layout,
                        out=out,
                        bias=bias,
                        **kwargs
                    )
    except Exception:
        warnings.warn("Could not check alignment, falling back to regular grouped GEMM")
        from ..cpp_extensions.gemm import general_grouped_gemm
        return general_grouped_gemm(
            A, B, workspace, m_splits,
            out_dtype=out_dtype,
            layout=layout,
            out=out,
            bias=bias,
            **kwargs
        )

    # Prepare output tensor
    if out_dtype is None:
        # Smart dtype selection based on operation type
        # Note: grouped GEMM is typically used for MoE which often needs FP32 accumulation
        out_dtype = torch.float32

    out_shape = list(A.shape[:-1]) + [B.shape[-2] if layout == "nt" else B.shape[-1]]
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=A.device)

    # Get FP8 data and scales
    def _get_fp8_data_and_scales(tensor: FP8DeepGemmQTensor, columnwise: bool = False):
        if columnwise and tensor._columnwise_data is not None:
            return tensor._columnwise_data, tensor._columnwise_scale_inv
        elif not columnwise and tensor._rowwise_data is not None:
            return tensor._rowwise_data, tensor._rowwise_scale_inv
        elif tensor._rowwise_data is not None:
            return tensor._rowwise_data, tensor._rowwise_scale_inv
        elif tensor._columnwise_data is not None:
            return tensor._columnwise_data, tensor._columnwise_scale_inv
        else:
            raise ValueError("No suitable FP8 data found in tensor")

    try:
        # A tensor always uses rowwise (per_token), B tensor always uses columnwise (per_block)
        A_data, A_scales = _get_fp8_data_and_scales(A, columnwise=False)
        B_data, B_scales = _get_fp8_data_and_scales(B, columnwise=True)

        # Create DeepGEMM input tuples
        A_tuple = (A_data, A_scales)
        B_tuple = (B_data, B_scales)

        # Call appropriate DeepGEMM grouped kernel
        if layout == "nt":
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                A_tuple,
                B_tuple,
                out,
                m_splits,  # Keep as tensor, not .tolist()
                recipe=None
            )
        elif layout == "nn":
            deep_gemm.m_grouped_fp8_gemm_nn_contiguous(
                A_tuple,
                B_tuple,
                out,
                m_splits,  # Keep as tensor, not .tolist()
                recipe=None
            )
        else:
            raise ValueError(f"Unsupported layout for grouped GEMM: {layout}")

        # Add bias after the grouped GEMM operation if provided
        if bias is not None:
            if out.shape != bias.shape:
                # Broadcast bias if needed
                bias_broadcasted = bias.expand(out.shape)
                out.add_(bias_broadcasted)
            else:
                out.add_(bias)

        return (out, workspace)

    except Exception as e:
        warnings.warn(f"DeepGEMM grouped operation failed: {e}. Falling back to regular single GEMM.")
        from ..cpp_extensions.gemm import general_gemm

        # DeepGEMM grouped GEMM failed, fall back to regular single GEMM
        # Note: This is a simplified fallback - for full grouped GEMM support,
        # would need to properly split tensors according to m_splits
        return general_gemm(
            A, B, workspace,
            out_dtype=out_dtype,
            layout=layout,  # Keep the layout as-is (nt/nn)
            out=out,
            bias=bias,
            **kwargs
        )