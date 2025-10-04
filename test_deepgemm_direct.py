#!/usr/bin/env python3

"""Test DeepGEMM directly to understand its expected input format"""

import torch
import warnings
warnings.simplefilter("always")

def test_deepgemm_direct():
    print("Testing DeepGEMM directly")
    print("=" * 40)

    try:
        import deep_gemm
        print("✓ DeepGEMM imported successfully")

        # Test what DeepGEMM expects for NT layout
        device = torch.device('cuda')

        # Use simple aligned dimensions
        M, K, N = 128, 128, 128  # All aligned to 128

        print(f"Testing with dimensions: M={M}, K={K}, N={N}")

        # Create FP8 tensors directly like in DeepGEMM tests
        a_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        b_bf16 = torch.randn(K, N, device=device, dtype=torch.bfloat16)  # NT layout: B should be K x N

        print(f"A shape: {a_bf16.shape}")
        print(f"B shape: {b_bf16.shape}")

        # Convert to FP8 using DeepGEMM's utilities
        from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

        a_fp8 = per_token_cast_to_fp8(a_bf16, use_ue8m0=False)
        b_fp8 = per_block_cast_to_fp8(b_bf16, use_ue8m0=False)

        print(f"A FP8 data shape: {a_fp8[0].shape}")
        print(f"A FP8 scales shape: {a_fp8[1].shape}")
        print(f"B FP8 data shape: {b_fp8[0].shape}")
        print(f"B FP8 scales shape: {b_fp8[1].shape}")

        # Create output tensor
        d = torch.empty(M, N, device=device, dtype=torch.bfloat16)

        print(f"Output shape: {d.shape}")

        # Try calling DeepGEMM NT directly
        print("\nCalling deep_gemm.fp8_gemm_nt...")
        deep_gemm.fp8_gemm_nt(
            a_fp8,
            b_fp8,
            d,
            c=None,
            disable_ue8m0_cast=True,
            recipe=None
        )

        print("✅ DeepGEMM NT call succeeded!")
        print(f"Result range: [{d.min().item():.3f}, {d.max().item():.3f}]")

        return True

    except Exception as e:
        print(f"❌ DeepGEMM direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_deepgemm_direct()
    if success:
        print("\n✅ DeepGEMM direct test PASSED!")
    else:
        print("\n❌ DeepGEMM direct test FAILED!")