#!/usr/bin/env python3

"""Check DeepGEMM operations and accumulation support"""

import torch

def check_deepgemm_ops():
    print("DeepGEMM Operations and Accumulation Support")
    print("=" * 50)

    try:
        import deep_gemm

        # List all available DeepGEMM operations
        print("Available DeepGEMM operations:")
        deepgemm_ops = [attr for attr in dir(deep_gemm) if 'gemm' in attr.lower() and not attr.startswith('_')]
        for op in sorted(deepgemm_ops):
            print(f"  - {op}")

        print("\nTesting accumulation patterns...")

        device = torch.device('cuda')
        M, K, N = 128, 128, 128

        # Create test data like in working example
        from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

        A_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B_bf16 = torch.randn(K, N, device=device, dtype=torch.bfloat16)

        A_fp8 = per_token_cast_to_fp8(A_bf16, use_ue8m0=False)
        B_fp8 = per_block_cast_to_fp8(B_bf16, use_ue8m0=False)

        print(f"A_fp8: data={A_fp8[0].shape}, scales={A_fp8[1].shape}")
        print(f"B_fp8: data={B_fp8[0].shape}, scales={B_fp8[1].shape}")

        # Test different output dtypes and accumulation
        test_cases = [
            ("bfloat16", torch.bfloat16, False, "Forward GEMM (no accumulation)"),
            ("bfloat16", torch.bfloat16, True, "Forward GEMM (with bias/accumulation)"),
            ("float32", torch.float32, False, "Backward GEMM (no accumulation)"),
            ("float32", torch.float32, True, "Backward GEMM (with accumulation)")
        ]

        for dtype_name, dtype, accumulate, description in test_cases:
            print(f"\n--- {description} ---")

            # Create output tensor
            d = torch.empty(M, N, device=device, dtype=dtype)

            # Create bias/accumulation tensor if needed
            c = None
            if accumulate:
                c = torch.randn(M, N, device=device, dtype=dtype) * 0.1
                d.copy_(c)  # Pre-fill with values for accumulation

            try:
                # Test DeepGEMM call
                deep_gemm.fp8_gemm_nt(
                    A_fp8,
                    B_fp8,
                    d,
                    c=c,
                    disable_ue8m0_cast=True,
                    recipe=None
                )

                print(f"  ✅ Success: {dtype_name} output with accumulate={accumulate}")
                print(f"  Result range: [{d.min().item():.6f}, {d.max().item():.6f}]")

            except Exception as e:
                print(f"  ❌ Failed: {dtype_name} output with accumulate={accumulate}")
                print(f"  Error: {e}")

        # Test in-place accumulation patterns
        print(f"\n--- Testing In-Place FP32 Accumulation ---")

        # Create FP32 accumulation tensor
        accumulator = torch.randn(M, N, device=device, dtype=torch.float32) * 0.01
        original_accumulator = accumulator.clone()

        try:
            # Test if DeepGEMM can accumulate directly into FP32 tensor
            deep_gemm.fp8_gemm_nt(
                A_fp8,
                B_fp8,
                accumulator,  # Output is FP32
                c=original_accumulator,  # Accumulate with original values
                disable_ue8m0_cast=True,
                recipe=None
            )

            print("  ✅ In-place FP32 accumulation works")
            print(f"  Result range: [{accumulator.min().item():.6f}, {accumulator.max().item():.6f}]")

            # Verify accumulation happened
            diff = (accumulator - original_accumulator).abs().max()
            print(f"  Max change from original: {diff.item():.6f}")

        except Exception as e:
            print(f"  ❌ In-place FP32 accumulation failed: {e}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_deepgemm_ops()