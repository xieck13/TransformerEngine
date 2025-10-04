#!/usr/bin/env python3

"""Debug script to understand the tensor shape mismatch issue"""

import torch
import warnings
warnings.simplefilter("always")

def debug_quantizer_shapes():
    print("Debugging FP8DeepGemmQuantizer Shape Issues")
    print("=" * 50)

    try:
        import transformer_engine.pytorch
        import transformer_engine_torch as tex
        from transformer_engine_torch import DType as TE_DType
        from transformer_engine.pytorch.tensor.float8_deepgemm_tensor import FP8DeepGemmQuantizer

        device = torch.device('cuda')

        # Test the problematic size: (256, 512, 384)
        M, K, N = 256, 512, 384

        print(f"Testing size: A=({M}, {K}), B=({K}, {N})")

        # Create quantizers
        A_quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=True, columnwise=False, use_deepgemm_layout=True
        )
        B_quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=False, columnwise=True, use_deepgemm_layout=True
        )

        # Create tensors
        A_tensor = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B_tensor = torch.randn(K, N, device=device, dtype=torch.bfloat16)

        print(f"A_tensor.shape: {A_tensor.shape}")
        print(f"B_tensor.shape: {B_tensor.shape}")

        # Check what shapes the quantizers expect
        A_scale_shape = A_quantizer.get_scale_shape(A_tensor.shape, columnwise=False)
        B_scale_shape = B_quantizer.get_scale_shape(B_tensor.shape, columnwise=True)

        print(f"A_scale_shape (rowwise): {A_scale_shape}")
        print(f"B_scale_shape (columnwise): {B_scale_shape}")

        # Create quantized tensors
        A_quantized = A_quantizer.make_empty(A_tensor.shape, dtype=A_tensor.dtype, device=device)
        B_quantized = B_quantizer.make_empty(B_tensor.shape, dtype=B_tensor.dtype, device=device)

        print(f"A_quantized._rowwise_data: {A_quantized._rowwise_data.shape if A_quantized._rowwise_data is not None else None}")
        print(f"A_quantized._rowwise_scale_inv: {A_quantized._rowwise_scale_inv.shape if A_quantized._rowwise_scale_inv is not None else None}")
        print(f"B_quantized._columnwise_data: {B_quantized._columnwise_data.shape if B_quantized._columnwise_data is not None else None}")
        print(f"B_quantized._columnwise_scale_inv: {B_quantized._columnwise_scale_inv.shape if B_quantized._columnwise_scale_inv is not None else None}")

        # Test the actual quantization functions from deep_gemm
        try:
            from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

            print("\nTesting deep_gemm utility functions:")
            A_fp8_data, A_fp8_scales = per_token_cast_to_fp8(A_tensor, use_ue8m0=False)
            B_fp8_data, B_fp8_scales = per_block_cast_to_fp8(B_tensor, use_ue8m0=False)

            print(f"per_token_cast_to_fp8(A): data={A_fp8_data.shape}, scales={A_fp8_scales.shape}")
            print(f"per_block_cast_to_fp8(B): data={B_fp8_data.shape}, scales={B_fp8_scales.shape}")

            # Check if there's a mismatch
            if B_fp8_data.shape != B_quantized._columnwise_data.shape:
                print(f"❌ MISMATCH: per_block_cast_to_fp8 returns {B_fp8_data.shape}, but quantized tensor expects {B_quantized._columnwise_data.shape}")
            else:
                print("✅ B data shapes match")

            if B_fp8_scales.shape != B_quantized._columnwise_scale_inv.shape:
                print(f"❌ MISMATCH: per_block_cast_to_fp8 scales {B_fp8_scales.shape}, but quantized tensor expects {B_quantized._columnwise_scale_inv.shape}")
            else:
                print("✅ B scale shapes match")

        except ImportError as e:
            print(f"Could not import deep_gemm utils: {e}")

        # Try the actual quantization
        print(f"\nTrying quantization...")
        try:
            A_quantizer.update_quantized(A_tensor, A_quantized)
            print("✅ A quantization successful")
        except Exception as e:
            print(f"❌ A quantization failed: {e}")

        try:
            B_quantizer.update_quantized(B_tensor, B_quantized)
            print("✅ B quantization successful")
        except Exception as e:
            print(f"❌ B quantization failed: {e}")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_quantizer_shapes()