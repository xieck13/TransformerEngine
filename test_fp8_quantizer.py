#!/usr/bin/env python3

"""Test our FP8DeepGemmQuantizer with DeepGEMM utilities"""

import torch
import warnings
warnings.simplefilter("always")

def test_fp8_quantizer():
    print("Testing FP8DeepGemmQuantizer")
    print("=" * 40)

    try:
        # Import all required modules
        import transformer_engine.pytorch
        import transformer_engine_torch as tex
        from transformer_engine_torch import DType as TE_DType

        from transformer_engine.pytorch.tensor.float8_deepgemm_tensor import FP8DeepGemmQuantizer
        print("✓ All imports successful")

        # Create quantizer
        device = torch.device('cuda')
        quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,  # Keep it simple for now
            use_deepgemm_layout=True
        )
        print("✓ FP8DeepGemmQuantizer created")

        # Create test tensor
        M, K = 128, 128
        test_tensor = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        print(f"✓ Created test tensor: {test_tensor.shape}")

        # Create quantized tensor
        quantized_tensor = quantizer.make_empty(test_tensor.shape, dtype=test_tensor.dtype, device=device)
        print(f"✓ Created quantized tensor container")

        # Check the data types
        print(f"  Rowwise data dtype: {quantized_tensor._rowwise_data.dtype}")
        print(f"  Rowwise scales dtype: {quantized_tensor._rowwise_scale_inv.dtype}")
        print(f"  Rowwise data shape: {quantized_tensor._rowwise_data.shape}")
        print(f"  Rowwise scales shape: {quantized_tensor._rowwise_scale_inv.shape}")

        # Try quantization
        print("\n" + "=" * 30)
        print("Testing quantization...")
        quantizer.update_quantized(test_tensor, quantized_tensor)
        print("✓ Quantization successful!")

        # Check if we can use this with DeepGEMM
        print("\n" + "=" * 30)
        print("Testing DeepGEMM compatibility...")

        import deep_gemm

        # Create another quantized tensor for B
        B_tensor = torch.randn(K, M, device=device, dtype=torch.bfloat16)  # K x M for NT
        B_quantized = quantizer.make_empty(B_tensor.shape, dtype=B_tensor.dtype, device=device)
        quantizer.update_quantized(B_tensor, B_quantized)

        # Extract tuples
        A_tuple = (quantized_tensor._rowwise_data, quantized_tensor._rowwise_scale_inv)
        B_tuple = (B_quantized._rowwise_data, B_quantized._rowwise_scale_inv)

        print(f"A tuple shapes: {A_tuple[0].shape}, {A_tuple[1].shape}")
        print(f"B tuple shapes: {B_tuple[0].shape}, {B_tuple[1].shape}")
        print(f"A data dtype: {A_tuple[0].dtype}")
        print(f"B data dtype: {B_tuple[0].dtype}")

        # Create output tensor
        output = torch.empty(M, M, device=device, dtype=torch.bfloat16)

        # Try DeepGEMM call
        deep_gemm.fp8_gemm_nt(
            A_tuple,
            B_tuple,
            output,
            c=None,
            disable_ue8m0_cast=True,
            recipe=None
        )

        print("✅ DeepGEMM call with our quantizer succeeded!")
        print(f"Result range: [{output.min().item():.3f}, {output.max().item():.3f}]")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fp8_quantizer()
    if success:
        print("\n✅ FP8DeepGemmQuantizer test PASSED!")
    else:
        print("\n❌ FP8DeepGemmQuantizer test FAILED!")