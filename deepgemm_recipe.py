# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DeepGEMM-optimized FP8 recipe for enhanced precision and performance."""

import os
from typing import Optional
from dataclasses import dataclass
from transformer_engine.common.recipe import Recipe, Format, QParams, MMParams
from transformer_engine_torch import DType as TE_DType


@dataclass()
class DeepGEMMFP8Recipe(Recipe):
    """
    FP8 recipe optimized for DeepGEMM operations.

    This recipe is specifically designed to work with DeepGEMM-based modules
    (LinearDeepGemm, LayerNormLinearDeepGemm, GroupedLinearDeepGemm) to provide
    optimal performance and precision using native DeepGEMM FP8 kernels.

    Key Features:
    - Block-wise scaling optimized for DeepGEMM layout requirements
    - 1D1D kernel preference for weight gradients with fp32 accumulation
    - Dimension constraints aligned with DeepGEMM requirements (multiples of 128)
    - Enhanced precision through DeepGEMM's optimized quantization strategies
    - Megatron-LM main_grad compatibility for fp32 accumulation

    Parameters
    ----------
    fp8_format : Format, default = Format.E4M3
                Controls the FP8 data format. E4M3 is recommended for DeepGEMM
                operations to maximize precision while maintaining performance.
    use_deepgemm_layout : bool, default = True
                        Whether to use DeepGEMM-optimized tensor layouts for
                        quantization. This enables block-wise scaling patterns
                        that align with DeepGEMM kernel requirements.
    enable_1d1d_wgrad : bool, default = True
                       Enable 1D1D kernel preference for weight gradients.
                       This provides better precision compared to 1D2D kernels.
    fp32_accumulation : bool, default = True
                       Enable fp32 accumulation for weight gradients. This
                       eliminates precision loss compared to bf16 accumulation.
    block_scaling_dim : int, default = 2
                       Block scaling dimension for quantization. 2D scaling
                       provides better precision for larger matrices.
    enforce_dim_constraints : bool, default = True
                            Enforce DeepGEMM dimension constraints (multiples
                            of 128). When True, operations will fail fast if
                            tensor dimensions don't meet DeepGEMM requirements.
    margin : int, default = 0
            Margin for scaling factor computation. 0 provides maximum
            utilization of FP8 dynamic range.
    power_2_scales : bool, default = False
                    Whether to constrain scaling factors to powers of 2.
                    DeepGEMM works well with arbitrary float32 scales.

    Usage Example
    -------------
    ```python
    import torch
    from transformer_engine.pytorch import fp8_autocast
    from transformer_engine.pytorch.module.linear_deepgemm import LinearDeepGemm
    from deepgemm_recipe import DeepGEMMFP8Recipe

    # Create DeepGEMM-optimized recipe
    recipe = DeepGEMMFP8Recipe(
        fp8_format=Format.E4M3,
        enable_1d1d_wgrad=True,
        fp32_accumulation=True,
        enforce_dim_constraints=True
    )

    # Use with DeepGEMM modules
    model = LinearDeepGemm(
        in_features=4096,
        out_features=4096,
        accumulate_into_main_grad=True,  # Enable fp32 main_grad
        use_deepgemm=True
    )

    # Enable FP8 training with DeepGEMM optimization
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        input_tensor = torch.randn(128, 4096, device='cuda', dtype=torch.bfloat16)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()  # Uses 1D1D wgrad with fp32 accumulation
    ```

    Performance Notes
    -----------------
    - All tensor dimensions should be multiples of 128 for optimal performance
    - 1D1D wgrad kernels provide better precision than 1D2D at similar performance
    - fp32 accumulation eliminates precision loss with minimal overhead
    - Block-wise scaling reduces quantization error compared to per-tensor scaling

    Compatibility
    -------------
    - LinearDeepGemm: Full support with 1D1D wgrad and fp32 accumulation
    - LayerNormLinearDeepGemm: Full support with fused operations
    - GroupedLinearDeepGemm: Full support with individual GEMM optimization
    - Regular TE modules: Falls back to standard FP8 operations
    """

    # Core FP8 configuration
    fp8_format: Format = Format.E4M3
    margin: int = 0

    # DeepGEMM-specific configuration
    use_deepgemm_layout: bool = True
    enable_1d1d_wgrad: bool = True
    fp32_accumulation: bool = True
    block_scaling_dim: int = 2
    enforce_dim_constraints: bool = True

    # Advanced configuration
    power_2_scales: bool = False
    use_f32_scales: bool = True  # DeepGEMM works well with fp32 scales

    # Quantization parameters optimized for DeepGEMM
    fp8_quant_fwd_inp: QParams = QParams(
        power_2_scale=False,  # Use fp32 scales for better precision
        amax_epsilon=1e-10,   # Small epsilon for numerical stability
        random_hadamard_transform=False,  # Not needed for DeepGEMM
        stochastic_rounding=False  # Deterministic quantization
    )

    fp8_quant_fwd_weight: QParams = QParams(
        power_2_scale=False,
        amax_epsilon=1e-10,
        random_hadamard_transform=False,
        stochastic_rounding=False
    )

    fp8_quant_bwd_grad: QParams = QParams(
        power_2_scale=False,
        amax_epsilon=1e-10,
        random_hadamard_transform=False,
        stochastic_rounding=False
    )

    # GEMM parameters optimized for DeepGEMM kernels
    fp8_gemm_fprop: MMParams = MMParams(use_split_accumulator=False)  # DeepGEMM handles accumulation
    fp8_gemm_dgrad: MMParams = MMParams(use_split_accumulator=False)  # Native DeepGEMM NT layout
    fp8_gemm_wgrad: MMParams = MMParams(use_split_accumulator=False)  # 1D1D with fp32 accumulation

    # Attention configuration (future extension)
    fp8_dpa: bool = False  # Not yet supported with DeepGEMM
    fp8_mha: bool = False  # Not yet supported with DeepGEMM

    def __post_init__(self) -> None:
        """Validate recipe configuration."""
        assert self.fp8_format in [Format.E4M3, Format.HYBRID], \
            "DeepGEMM recipe only supports E4M3 or HYBRID formats"

        assert self.block_scaling_dim in [1, 2], \
            "Block scaling dimension must be 1 or 2"

        if self.enforce_dim_constraints:
            # This will be checked at runtime by DeepGEMM modules
            pass

    def get_deepgemm_config(self) -> dict:
        """
        Get DeepGEMM-specific configuration dictionary.

        Returns
        -------
        dict
            Configuration dictionary for DeepGEMM modules
        """
        return {
            'fp8_dtype': TE_DType.kFloat8E4M3 if self.fp8_format == Format.E4M3 else TE_DType.kFloat8E4M3,
            'block_scaling_dim': self.block_scaling_dim,
            'use_deepgemm_layout': self.use_deepgemm_layout,
            'accumulate_into_main_grad': self.fp32_accumulation,
            'enforce_dim_constraints': self.enforce_dim_constraints,
        }

    def is_compatible_with_deepgemm(self) -> bool:
        """
        Check if this recipe is compatible with DeepGEMM operations.

        Returns
        -------
        bool
            True if compatible, False otherwise
        """
        return (
            self.use_deepgemm_layout and
            self.fp8_format in [Format.E4M3, Format.HYBRID] and
            not self.fp8_dpa and  # DeepGEMM attention not yet supported
            not self.fp8_mha
        )

    def get_quantizer_kwargs(self, role: str) -> dict:
        """
        Get quantizer keyword arguments for specific tensor roles.

        Parameters
        ----------
        role : str
            Tensor role (e.g., 'input', 'weight', 'grad_output')

        Returns
        -------
        dict
            Quantizer configuration for the given role
        """
        base_kwargs = {
            'fp8_dtype': TE_DType.kFloat8E4M3 if self.fp8_format == Format.E4M3 else TE_DType.kFloat8E4M3,
            'use_deepgemm_layout': self.use_deepgemm_layout,
            'block_scaling_dim': self.block_scaling_dim,
        }

        if role in ['input', 'grad_output']:
            base_kwargs.update({
                'rowwise': True,
                'columnwise': False,
            })
        elif role == 'weight':
            base_kwargs.update({
                'rowwise': True,
                'columnwise': True,
            })

        return base_kwargs

    def __repr__(self) -> str:
        """String representation of the recipe."""
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"use_deepgemm_layout={self.use_deepgemm_layout}, "
            f"enable_1d1d_wgrad={self.enable_1d1d_wgrad}, "
            f"fp32_accumulation={self.fp32_accumulation}, "
            f"block_scaling_dim={self.block_scaling_dim}, "
            f"enforce_dim_constraints={self.enforce_dim_constraints}, "
            f"margin={self.margin}"
        )


# Predefined recipe configurations
def deepgemm_precision_recipe() -> DeepGEMMFP8Recipe:
    """
    Precision-optimized DeepGEMM recipe.

    This configuration prioritizes numerical accuracy over performance,
    using fp32 accumulation and strict dimension constraints.
    """
    return DeepGEMMFP8Recipe(
        fp8_format=Format.E4M3,
        enable_1d1d_wgrad=True,
        fp32_accumulation=True,
        block_scaling_dim=2,
        enforce_dim_constraints=True,
        margin=2,  # More conservative scaling for better precision
        use_f32_scales=True
    )


def deepgemm_performance_recipe() -> DeepGEMMFP8Recipe:
    """
    Performance-optimized DeepGEMM recipe.

    This configuration prioritizes performance while maintaining
    acceptable precision for most use cases.
    """
    return DeepGEMMFP8Recipe(
        fp8_format=Format.E4M3,
        enable_1d1d_wgrad=True,
        fp32_accumulation=True,  # Still use fp32 for gradient accumulation
        block_scaling_dim=1,     # Simpler scaling for speed
        enforce_dim_constraints=False,  # More flexible dimension handling
        margin=0,  # Maximum FP8 range utilization
        power_2_scales=True  # Faster scale computation
    )


def deepgemm_megatron_recipe() -> DeepGEMMFP8Recipe:
    """
    Megatron-LM optimized DeepGEMM recipe.

    This configuration is optimized for Megatron-LM training with
    main_grad accumulation and tensor parallelism support.
    """
    return DeepGEMMFP8Recipe(
        fp8_format=Format.HYBRID,  # Better for mixed precision training
        enable_1d1d_wgrad=True,
        fp32_accumulation=True,  # Essential for main_grad precision
        block_scaling_dim=2,
        enforce_dim_constraints=True,  # Strict constraints for TP
        margin=1,  # Balanced precision/range tradeoff
        use_f32_scales=True
    )