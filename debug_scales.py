#!/usr/bin/env python3

"""Debug script to understand DeepGEMM scale tensor shapes"""

import torch

def debug_deepgemm_scales():
    print("Debugging DeepGEMM Scale Tensor Shapes")
    print("=" * 50)

    try:
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

        device = torch.device('cuda')

        # Test various tensor sizes
        test_cases = [
            (128, 128),
            (256, 512),
            (64, 256),
        ]

        for M, K in test_cases:
            print(f"\nTesting M={M}, K={K}")
            print("-" * 30)

            # Create test tensor
            test_tensor = torch.randn(M, K, device=device, dtype=torch.bfloat16)
            print(f"Input tensor shape: {test_tensor.shape}")

            # Test per_token_cast_to_fp8
            fp8_data, fp8_scales = per_token_cast_to_fp8(test_tensor, use_ue8m0=False)
            print(f"per_token_cast_to_fp8:")
            print(f"  Data shape: {fp8_data.shape}")
            print(f"  Scales shape: {fp8_scales.shape}")
            print(f"  Scales dims: {fp8_scales.dim()}")

            # Test per_block_cast_to_fp8
            fp8_data_block, fp8_scales_block = per_block_cast_to_fp8(test_tensor, use_ue8m0=False)
            print(f"per_block_cast_to_fp8:")
            print(f"  Data shape: {fp8_data_block.shape}")
            print(f"  Scales shape: {fp8_scales_block.shape}")
            print(f"  Scales dims: {fp8_scales_block.dim()}")

            # Test with transposed tensor (for B in NT layout)
            # Note: per_token_cast_to_fp8 requires K dimension to be multiple of 128
            if M % 128 == 0:  # Only test if dimension is compatible
                test_tensor_T = torch.randn(K, M, device=device, dtype=torch.bfloat16)
                print(f"\nTransposed tensor shape: {test_tensor_T.shape}")

                try:
                    fp8_data_T, fp8_scales_T = per_token_cast_to_fp8(test_tensor_T, use_ue8m0=False)
                    print(f"per_token_cast_to_fp8 (transposed):")
                    print(f"  Data shape: {fp8_data_T.shape}")
                    print(f"  Scales shape: {fp8_scales_T.shape}")
                except AssertionError as e:
                    print(f"per_token_cast_to_fp8 (transposed): Assertion error - {e}")

                fp8_data_block_T, fp8_scales_block_T = per_block_cast_to_fp8(test_tensor_T, use_ue8m0=False)
                print(f"per_block_cast_to_fp8 (transposed):")
                print(f"  Data shape: {fp8_data_block_T.shape}")
                print(f"  Scales shape: {fp8_scales_block_T.shape}")
            else:
                print(f"\nSkipping transposed test for M={M} (not multiple of 128)")

        print("\n" + "=" * 50)
        print("Key Observations:")
        print("- per_token_cast_to_fp8 produces scales with shape [M, ceil(K/128)] for input [M, K]")
        print("- per_block_cast_to_fp8 produces scales with shape [ceil(M/128), ceil(K/128)] for input [M, K]")
        print("- per_token_cast_to_fp8 requires K dimension to be multiple of 128")
        print("- For NT layout: A[M,K] uses per_token, B[K,N] uses per_block")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_deepgemm_scales()