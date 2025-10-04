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
        # Determine if this is an accumulation operation that needs 1D1D kernel
        # 1D1D kernel (accumulation): Use when we have accumulate=True
        # 1D2D kernel (no accumulation): Use for simple forward pass operations
        needs_accumulation_in_kernel = accumulate

        print(f"DEBUG: accumulate={accumulate}, bias={bias is not None}, needs_accumulation_in_kernel={needs_accumulation_in_kernel}")

        if needs_accumulation_in_kernel:
            # 1D1D kernel: in-place accumulation, requires float32 output
            out_dtype = torch.float32
            print(f"DEBUG: Using 1D1D kernel with float32 output")
        else:
            # 1D2D kernel: simple GEMM, uses bfloat16 output
            out_dtype = torch.bfloat16
            print(f"DEBUG: Using 1D2D kernel with bfloat16 output")

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

    # Prepare bias handling FIRST (needed for kernel selection)
    # For 1D1D kernel (in-place accumulation): accumulate=True means we accumulate into output
    # The c_tensor should be the output tensor itself for in-place accumulation
    needs_accumulation_in_kernel = accumulate

    if needs_accumulation_in_kernel:
        # 1D1D kernel: pass output tensor as c_tensor for in-place accumulation
        # Pre-initialize output with bias if provided
        if bias is not None:
            if out.shape != bias.shape:
                # Broadcast bias if needed
                bias_broadcasted = bias.expand(out.shape)
                out.copy_(bias_broadcasted)
            else:
                out.copy_(bias)
        c_tensor = out  # Use output tensor for in-place accumulation
        bias_to_add_later = None  # Don't add bias later, it's already in the output
    else:
        # 1D2D kernel: add bias after DeepGEMM operation
        c_tensor = None  # Pass None to DeepGEMM 1D2D kernel
        bias_to_add_later = bias  # Add bias later if provided

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

        # B tensor quantization depends on kernel type:
        # - 1D1D kernels (accumulation): need per_token quantization for B (rowwise)
        # - 1D2D kernels (no accumulation): need per_block quantization for B (columnwise)
        if needs_accumulation_in_kernel:
            # 1D1D kernel: B should use per_token quantization (rowwise)
            if B._rowwise_data is not None:
                B_data, B_scales = B._rowwise_data, B._rowwise_scale_inv
                print(f"DEBUG: Using rowwise B data for 1D1D kernel")
            else:
                # If rowwise not available, this is an error for 1D1D kernels
                raise RuntimeError("1D1D kernel requires B tensor to have rowwise (per_token) quantization, "
                                 "but only columnwise data is available. Please quantize B tensor with rowwise=True.")
        else:
            # 1D2D kernel: B uses per_block quantization (columnwise)
            B_data, B_scales = _get_fp8_data_and_scales(B, columnwise=True)
            print(f"DEBUG: Using columnwise B data for 1D2D kernel")

        # Create DeepGEMM input tuples
        # DeepGEMM expects (data, scales) tuples directly
        A_tuple = (A_data, A_scales)
        B_tuple = (B_data, B_scales)

    except Exception as e:
        # Rather than complex fallback logic, fail cleanly with clear error message
        raise RuntimeError(f"Failed to prepare DeepGEMM data: {e}. "
                          f"FP8DeepGemmQTensor objects are only compatible with DeepGEMM operations. "
                          f"Use regular Float8BlockwiseQTensor with general_gemm for fallback capability.")

    # Note: accumulate=True without bias means accumulate into output tensor,
    # but DeepGEMM doesn't support this directly - we handle it with alpha/beta scaling

    try:
        # Choose the correct recipe based on operation type
        # Recipe format: (dim1, dim2, dim3) where dim2 controls kernel selection:
        # - dim2 == 1: 1D1D kernel (supports accumulation, requires float32, c!=None)
        # - dim2 != 1: 1D2D kernel (no accumulation, requires bfloat16, c=None)
        if needs_accumulation_in_kernel:
            # Use 1D1D kernel for accumulation operations
            recipe = (1, 1, 128)
            print(f"DEBUG: Selected 1D1D recipe {recipe}, c_tensor={c_tensor is not None}")
        else:
            # Use 1D2D kernel for simple GEMM operations
            recipe = (1, 128, 128)
            print(f"DEBUG: Selected 1D2D recipe {recipe}, c_tensor={c_tensor is not None}")

        # Call appropriate DeepGEMM kernel using the correct (data, scales) tuple format
        if layout == "nt":
            deep_gemm.fp8_gemm_nt(
                A_tuple,
                B_tuple,
                out,
                c=c_tensor,
                disable_ue8m0_cast=True,
                recipe=recipe
            )
        elif layout == "nn":
            deep_gemm.fp8_gemm_nn(
                A_tuple,
                B_tuple,
                out,
                c=c_tensor,
                disable_ue8m0_cast=True,
                recipe=recipe
            )
        elif layout == "tn":
            deep_gemm.fp8_gemm_tn(
                A_tuple,
                B_tuple,
                out,
                c=c_tensor,
                disable_ue8m0_cast=True,
                recipe=recipe
            )
        elif layout == "tt":
            deep_gemm.fp8_gemm_tt(
                A_tuple,
                B_tuple,
                out,
                c=c_tensor,
                disable_ue8m0_cast=True,
                recipe=recipe
            )
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        # Apply scaling factors and bias after DeepGEMM operation
        if alpha != 1.0:
            out.mul_(alpha)

        # Add bias if provided (since we didn't pass it to DeepGEMM)
        if bias_to_add_later is not None:
            if out.shape != bias_to_add_later.shape:
                # Broadcast bias if needed
                bias_broadcasted = bias_to_add_later.expand(out.shape)
                out.add_(bias_broadcasted)
            else:
                out.add_(bias_to_add_later)

        # Handle beta scaling (typically used with accumulate=True)
        if beta is not None and beta != 1.0:
            out.mul_(beta)

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
                m_splits.tolist(),
                c=bias,
                disable_ue8m0_cast=True,
                recipe=None
            )
        elif layout == "nn":
            deep_gemm.m_grouped_fp8_gemm_nn_contiguous(
                A_tuple,
                B_tuple,
                out,
                m_splits.tolist(),
                c=bias,
                disable_ue8m0_cast=True,
                recipe=None
            )
        else:
            raise ValueError(f"Unsupported layout for grouped GEMM: {layout}")

        return (out, workspace)

    except Exception as e:
        warnings.warn(f"DeepGEMM grouped operation failed: {e}. Falling back to regular grouped GEMM.")
        from ..cpp_extensions.gemm import general_grouped_gemm

        # Remove out_dtype from kwargs if it exists to avoid duplicate argument
        kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'out_dtype'}

        return general_grouped_gemm(
            A, B, workspace, m_splits,
            out_dtype=out_dtype,
            layout=layout.upper(),  # Convert to uppercase for general_grouped_gemm
            out=out,
            bias=bias,
            **kwargs_filtered
        )