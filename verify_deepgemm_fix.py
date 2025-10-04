#!/usr/bin/env python3

"""Simple test to verify DeepGEMM is actually being called instead of falling back to general_gemm"""

import warnings
import sys
import os

# Capture warnings to see if we get fallback messages
warnings.simplefilter("always")

def test_deepgemm_actually_used():
    print("DeepGEMM Fix Verification Test")
    print("=" * 50)

    try:
        # Import all required modules
        import torch
        import transformer_engine.pytorch
        import transformer_engine_torch as tex
        from transformer_engine_torch import DType as TE_DType

        from transformer_engine.pytorch.tensor.float8_deepgemm_tensor import FP8DeepGemmQuantizer, FP8DeepGemmQTensor
        from transformer_engine.pytorch.cpp_extensions.deepgemm import deepgemm_fp8_gemm
        from transformer_engine.pytorch.utils import _empty_tensor

        print("✓ All imports successful")

        # Check if DeepGEMM is available
        try:
            import deep_gemm
            print("✓ DeepGEMM library available")
            deepgemm_available = True
        except ImportError:
            print("! DeepGEMM library not available - test will show fallback behavior")
            deepgemm_available = False

        # Create simple test tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("! CUDA not available - DeepGEMM requires CUDA")
            return False

        print(f"✓ Using device: {device}")

        # Create quantizers - different settings for A and B to match DeepGEMM expectations
        A_quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,  # A uses per_token (rowwise)
            use_deepgemm_layout=True
        )

        B_quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=False,
            columnwise=True,  # B uses per_block (columnwise)
            use_deepgemm_layout=True
        )
        print("✓ FP8DeepGemmQuantizers created")

        # Create test tensors with aligned dimensions (multiples of 128 for optimal DeepGEMM performance)
        M, K, N = 128, 128, 128  # Use smaller aligned dimensions for testing

        # Create input tensors - IMPORTANT: For NT layout, B should be (K, N)
        A_tensor = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B_tensor = torch.randn(K, N, device=device, dtype=torch.bfloat16)  # Correct: K x N for NT layout

        print(f"✓ Created test tensors: A({M}, {K}), B({K}, {N})")

        # Create quantized tensors using appropriate quantizers
        A_quantized = A_quantizer.make_empty(A_tensor.shape, dtype=A_tensor.dtype, device=device)
        B_quantized = B_quantizer.make_empty(B_tensor.shape, dtype=B_tensor.dtype, device=device)

        # Quantize the tensors
        A_quantizer.update_quantized(A_tensor, A_quantized)
        B_quantizer.update_quantized(B_tensor, B_quantized)

        print("✓ Tensors quantized successfully")

        # Create workspace
        workspace = _empty_tensor()

        # Capture warnings to see what happens
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call deepgemm_fp8_gemm
            print("\n" + "=" * 30)
            print("Calling deepgemm_fp8_gemm...")
            print("=" * 30)

            result, _ = deepgemm_fp8_gemm(
                A_quantized,
                B_quantized,
                workspace,
                layout="nt",
                out_dtype=torch.float32
            )

            print(f"✓ deepgemm_fp8_gemm completed successfully")
            print(f"  Result shape: {result.shape}")
            print(f"  Expected shape: ({M}, {N})")

            # Analyze warnings
            print(f"\nWarnings captured: {len(w)}")

            deepgemm_used = True
            for warning in w:
                warning_msg = str(warning.message)
                print(f"  Warning: {warning_msg}")

                # Check for fallback indicators
                if any(phrase in warning_msg.lower() for phrase in [
                    "falling back to regular gemm",
                    "deepgemm operation failed",
                    "failed to prepare deepgemm data",
                    "failed to transform scaling factors"
                ]):
                    deepgemm_used = False

            print("\n" + "=" * 50)
            if deepgemm_used and len(w) == 0:
                print("🎉 SUCCESS: DeepGEMM appears to be working correctly!")
                print("   No fallback warnings detected.")
            elif deepgemm_used and len(w) > 0:
                print("⚠️  PARTIAL SUCCESS: DeepGEMM called but with warnings.")
                print("   Check warnings above for optimization opportunities.")
            else:
                print("❌ FAILURE: DeepGEMM is still falling back to general_gemm.")
                print("   The fix didn't resolve the issue completely.")

            return deepgemm_used and len(w) == 0

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_deepgemm_actually_used()

    if success:
        print("\n✅ DeepGEMM fix verification PASSED!")
        sys.exit(0)
    else:
        print("\n❌ DeepGEMM fix verification FAILED!")
        print("\nNext steps:")
        print("1. Check DeepGEMM library installation")
        print("2. Ensure CUDA environment is available")
        print("3. Review the warning messages above")
        print("4. Consider testing with different tensor dimensions")
        sys.exit(1)