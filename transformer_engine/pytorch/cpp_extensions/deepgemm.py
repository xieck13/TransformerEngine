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

    # Prepare output tensor
    if out_dtype is None:
        out_dtype = torch.float32

    # Calculate output shape
    if layout in ["nt", "nn"]:
        out_shape = list(A.shape[:-1]) + [B.shape[-2] if layout == "nt" else B.shape[-1]]
    else:  # "tn", "tt"
        out_shape = list(A.shape[:-2]) + [A.shape[-1]] + [B.shape[-2] if layout == "tt" else B.shape[-1]]

    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=A.device)

    # Prepare FP8 data and scaling factors
    def _get_fp8_data_and_scales(tensor: FP8DeepGemmQTensor, columnwise: bool = False):
        if columnwise and tensor._columnwise_data is not None:
            return tensor._columnwise_data, tensor._columnwise_scale_inv
        elif tensor._rowwise_data is not None:
            return tensor._rowwise_data, tensor._rowwise_scale_inv
        else:
            raise ValueError("No suitable FP8 data found in tensor")

    # Get FP8 data and scales
    try:
        A_data, A_scales = _get_fp8_data_and_scales(A, columnwise=False)
        B_data, B_scales = _get_fp8_data_and_scales(B, columnwise=(layout in ["nt", "tt"]))

        # Create DeepGEMM input tuples
        # DeepGEMM expects (data, scales) tuples directly
        A_tuple = (A_data, A_scales)
        B_tuple = (B_data, B_scales)

    except Exception as e:
        warnings.warn(f"Failed to prepare DeepGEMM data: {e}")
        from ..cpp_extensions.gemm import general_gemm

        # Remove out_dtype from kwargs if it exists to avoid duplicate argument
        kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'out_dtype'}

        return general_gemm(
            A, B, workspace,
            out_dtype=out_dtype,
            layout=layout.upper(),  # Convert to uppercase for general_gemm
            out=out,
            bias=bias,
            accumulate=accumulate,
            alpha=alpha,
            beta=beta,
            **kwargs_filtered
        )

    # Prepare bias handling
    c_tensor = None
    if bias is not None:
        if out.shape != bias.shape:
            # Broadcast bias if needed
            c_tensor = bias.expand(out.shape).contiguous()
        else:
            c_tensor = bias
    elif accumulate:
        c_tensor = out

    try:
        # Call appropriate DeepGEMM kernel using the correct (data, scales) tuple format
        if layout == "nt":
            deep_gemm.fp8_gemm_nt(
                A_tuple,
                B_tuple,
                out,
                c=c_tensor,
                disable_ue8m0_cast=True,
                recipe=None
            )
        elif layout == "nn":
            deep_gemm.fp8_gemm_nn(
                A_tuple,
                B_tuple,
                out,
                c=c_tensor,
                disable_ue8m0_cast=True,
                recipe=None
            )
        elif layout == "tn":
            deep_gemm.fp8_gemm_tn(
                A_tuple,
                B_tuple,
                out,
                c=c_tensor,
                disable_ue8m0_cast=True,
                recipe=None
            )
        elif layout == "tt":
            deep_gemm.fp8_gemm_tt(
                A_tuple,
                B_tuple,
                out,
                c=c_tensor,
                disable_ue8m0_cast=True,
                recipe=None
            )
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        # Apply scaling factors
        if alpha != 1.0:
            out.mul_(alpha)
        if beta is not None and beta != 1.0 and c_tensor is not None:
            out.add_(c_tensor, alpha=beta - 1.0)

        # Return in format compatible with general_gemm
        return (out, workspace)

    except Exception as e:
        warnings.warn(f"DeepGEMM operation failed: {e}. Falling back to regular GEMM.")
        from ..cpp_extensions.gemm import general_gemm
        return general_gemm(
            A, B, workspace,
            out_dtype=out_dtype,
            layout=layout.upper(),  # Convert to uppercase for general_gemm
            out=out,
            bias=bias,
            accumulate=accumulate,
            alpha=alpha,
            beta=beta,
            **kwargs
        )


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
        out_dtype = torch.float32

    out_shape = list(A.shape[:-1]) + [B.shape[-2] if layout == "nt" else B.shape[-1]]
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=A.device)

    # Get FP8 data and scales
    def _get_fp8_data_and_scales(tensor: FP8DeepGemmQTensor, columnwise: bool = False):
        if columnwise and tensor._columnwise_data is not None:
            return tensor._columnwise_data, tensor._columnwise_scale_inv
        elif tensor._rowwise_data is not None:
            return tensor._rowwise_data, tensor._rowwise_scale_inv
        else:
            raise ValueError("No suitable FP8 data found in tensor")

    try:
        # Get FP8 data and scales
        A_data, A_scales = _get_fp8_data_and_scales(A, columnwise=False)
        B_data, B_scales = _get_fp8_data_and_scales(B, columnwise=(layout == "nt"))

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