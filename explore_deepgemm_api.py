#!/usr/bin/env python3

"""Exploration script to understand DeepGEMM API"""

import torch
import deep_gemm
import warnings

warnings.simplefilter("always")

def test_deepgemm_api():
    """Test various parameter combinations to understand correct usage"""
    device = torch.device('cuda')

    # Create test data using proper FP8 casting
    A_fp32 = torch.randn(128, 128, device=device, dtype=torch.float32)
    B_fp32 = torch.randn(128, 128, device=device, dtype=torch.float32)

    # Use DeepGEMM's own casting functions
    A_data, A_scales = deep_gemm.per_token_cast_to_fp8(A_fp32, torch.float8_e4m3fn)
    B_data_rowwise, B_scales_rowwise = deep_gemm.per_token_cast_to_fp8(B_fp32.t(), torch.float8_e4m3fn)
    B_data_columnwise, B_scales_columnwise = deep_gemm.per_block_cast_to_fp8(B_fp32, torch.float8_e4m3fn)

    print(f"A_data: {A_data.shape}, A_scales: {A_scales.shape}")
    print(f"B_data_rowwise: {B_data_rowwise.shape}, B_scales_rowwise: {B_scales_rowwise.shape}")
    print(f"B_data_columnwise: {B_data_columnwise.shape}, B_scales_columnwise: {B_scales_columnwise.shape}")

    # Test different output dtypes and parameter combinations
    output_dtypes = [torch.bfloat16, torch.float32]

    for out_dtype in output_dtypes:
        print(f"\n=== Testing with output dtype: {out_dtype} ===")
        output = torch.empty(128, 128, device=device, dtype=out_dtype)

        # Test 1: Basic call with rowwise B (likely 1D1D kernel)
        try:
            deep_gemm.fp8_gemm_nt(
                (A_data, A_scales),
                (B_data_rowwise, B_scales_rowwise),
                output
            )
            print(f"✓ Basic call with rowwise B works (dtype={out_dtype})")
        except Exception as e:
            print(f"✗ Basic call with rowwise B failed (dtype={out_dtype}): {e}")

        # Test 2: Basic call with columnwise B (likely 1D2D kernel)
        try:
            output = torch.empty(128, 128, device=device, dtype=out_dtype)
            deep_gemm.fp8_gemm_nt(
                (A_data, A_scales),
                (B_data_columnwise, B_scales_columnwise),
                output
            )
            print(f"✓ Basic call with columnwise B works (dtype={out_dtype})")
        except Exception as e:
            print(f"✗ Basic call with columnwise B failed (dtype={out_dtype}): {e}")

        # Test 3: With c=None (1D2D kernel expectation)
        try:
            output = torch.empty(128, 128, device=device, dtype=out_dtype)
            deep_gemm.fp8_gemm_nt(
                (A_data, A_scales),
                (B_data_columnwise, B_scales_columnwise),
                output,
                c=None
            )
            print(f"✓ Call with c=None works (dtype={out_dtype})")
        except Exception as e:
            print(f"✗ Call with c=None failed (dtype={out_dtype}): {e}")

        # Test 4: With c=output (1D1D kernel expectation)
        try:
            output = torch.empty(128, 128, device=device, dtype=out_dtype)
            deep_gemm.fp8_gemm_nt(
                (A_data, A_scales),
                (B_data_rowwise, B_scales_rowwise),
                output,
                c=output
            )
            print(f"✓ Call with c=output works (dtype={out_dtype})")
        except Exception as e:
            print(f"✗ Call with c=output failed (dtype={out_dtype}): {e}")

        # Test 5: With recipe=(1,1,128) (1D1D kernel)
        try:
            output = torch.empty(128, 128, device=device, dtype=out_dtype)
            deep_gemm.fp8_gemm_nt(
                (A_data, A_scales),
                (B_data_rowwise, B_scales_rowwise),
                output,
                recipe=(1, 1, 128)
            )
            print(f"✓ Call with recipe=(1,1,128) works (dtype={out_dtype})")
        except Exception as e:
            print(f"✗ Call with recipe=(1,1,128) failed (dtype={out_dtype}): {e}")

        # Test 6: With recipe=None (1D2D kernel)
        try:
            output = torch.empty(128, 128, device=device, dtype=out_dtype)
            deep_gemm.fp8_gemm_nt(
                (A_data, A_scales),
                (B_data_columnwise, B_scales_columnwise),
                output,
                recipe=None
            )
            print(f"✓ Call with recipe=None works (dtype={out_dtype})")
        except Exception as e:
            print(f"✗ Call with recipe=None failed (dtype={out_dtype}): {e}")

        # Test 7: Combination for 1D1D: rowwise B + c=output + recipe=(1,1,128)
        try:
            output = torch.empty(128, 128, device=device, dtype=out_dtype)
            deep_gemm.fp8_gemm_nt(
                (A_data, A_scales),
                (B_data_rowwise, B_scales_rowwise),
                output,
                c=output,
                recipe=(1, 1, 128)
            )
            print(f"✓ 1D1D combination works (dtype={out_dtype})")
        except Exception as e:
            print(f"✗ 1D1D combination failed (dtype={out_dtype}): {e}")

        # Test 8: Combination for 1D2D: columnwise B + c=None + recipe=None
        try:
            output = torch.empty(128, 128, device=device, dtype=out_dtype)
            deep_gemm.fp8_gemm_nt(
                (A_data, A_scales),
                (B_data_columnwise, B_scales_columnwise),
                output,
                c=None,
                recipe=None
            )
            print(f"✓ 1D2D combination works (dtype={out_dtype})")
        except Exception as e:
            print(f"✗ 1D2D combination failed (dtype={out_dtype}): {e}")

if __name__ == "__main__":
    test_deepgemm_api()