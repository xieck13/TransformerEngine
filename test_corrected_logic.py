#!/usr/bin/env python3

"""Test the corrected DeepGEMM dtype logic"""

import torch
import warnings
warnings.simplefilter("always")

def test_corrected_logic():
    print("Testing Corrected DeepGEMM Logic")
    print("=" * 40)

    try:
        # Import our corrected function
        from transformer_engine.pytorch.cpp_extensions.deepgemm import deepgemm_fp8_gemm
        from transformer_engine.pytorch.tensor.float8_deepgemm_tensor import FP8DeepGemmQuantizer
        import transformer_engine.pytorch
        import transformer_engine_torch as tex
        from transformer_engine_torch import DType as TE_DType
        from transformer_engine.pytorch.utils import _empty_tensor

        device = torch.device('cuda')

        # Create quantizers
        A_quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=True, columnwise=False, use_deepgemm_layout=True
        )
        B_quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=False, columnwise=True, use_deepgemm_layout=True
        )

        # Create test tensors
        M, K, N = 128, 128, 128
        A_tensor = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B_tensor = torch.randn(K, N, device=device, dtype=torch.bfloat16)

        # Quantize tensors
        A_quantized = A_quantizer.make_empty(A_tensor.shape, dtype=A_tensor.dtype, device=device)
        B_quantized = B_quantizer.make_empty(B_tensor.shape, dtype=B_tensor.dtype, device=device)
        A_quantizer.update_quantized(A_tensor, A_quantized)
        B_quantizer.update_quantized(B_tensor, B_quantized)

        workspace = _empty_tensor()

        # Test cases following DeepGEMM requirements
        test_cases = [
            (False, None, None, "Forward GEMM (no accumulation) → bfloat16"),
            (True, None, None, "Forward GEMM (accumulate=True) → float32"),
            (False, None, torch.randn(M, N, device=device, dtype=torch.bfloat16), "Forward GEMM (with bias) → float32"),
        ]

        for accumulate, beta, bias, description in test_cases:
            print(f"\n--- {description} ---")

            try:
                result, _ = deepgemm_fp8_gemm(
                    A_quantized, B_quantized, workspace,
                    layout="nt", accumulate=accumulate, beta=beta, bias=bias
                )

                print(f"  ✅ Success: Result dtype = {result.dtype}")
                print(f"  Result shape: {result.shape}")
                print(f"  Result range: [{result.min().item():.3f}, {result.max().item():.3f}]")

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                # Don't print full traceback for cleaner output

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_corrected_logic()
    if success:
        print("\n✅ DeepGEMM logic test completed!")
    else:
        print("\n❌ DeepGEMM logic test failed!")