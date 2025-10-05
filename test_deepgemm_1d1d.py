#!/usr/bin/env python3

"""Dedicated test for DeepGEMM 1D1D kernels"""

import torch
import warnings
import sys
warnings.simplefilter("always")

def test_deepgemm_1d1d():
    """Test DeepGEMM 1D1D kernel specifically"""
    print("DeepGEMM 1D1D Kernel Test Suite")
    print("=" * 50)

    try:
        # Import all required modules
        import transformer_engine.pytorch
        import transformer_engine_torch as tex
        from transformer_engine_torch import DType as TE_DType

        from transformer_engine.pytorch.tensor.float8_deepgemm_tensor import FP8DeepGemmQuantizer
        from transformer_engine.pytorch.cpp_extensions.deepgemm import deepgemm_fp8_gemm
        from transformer_engine.pytorch.utils import _empty_tensor
        import deep_gemm

        # Check CUDA availability
        if not torch.cuda.is_available():
            print("❌ CUDA not available - DeepGEMM requires CUDA")
            return False

        device = torch.device('cuda')
        print(f"✓ Using device: {device}")

        # Create quantizers with BOTH rowwise and columnwise support
        # A quantizer: rowwise (per_token) quantization
        A_quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=True, columnwise=False, use_deepgemm_layout=True
        )

        # B quantizer: BOTH rowwise and columnwise for maximum flexibility
        B_quantizer = FP8DeepGemmQuantizer(
            fp8_dtype=TE_DType.kFloat8E4M3,
            rowwise=True, columnwise=True, use_deepgemm_layout=True
        )
        print("✓ FP8DeepGemmQuantizers created with full quantization support")

        # Test matrix size
        M, K, N = 128, 128, 128
        print(f"\n--- Testing 1D1D kernels with size ({M}, {K}, {N}) ---")

        # Create test tensors
        A_tensor = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B_tensor = torch.randn(K, N, device=device, dtype=torch.bfloat16)

        # Create quantized tensors
        A_quantized = A_quantizer.make_empty(A_tensor.shape, dtype=A_tensor.dtype, device=device)
        B_quantized = B_quantizer.make_empty(B_tensor.shape, dtype=B_tensor.dtype, device=device)

        # Quantize tensors
        A_quantizer.update_quantized(A_tensor, A_quantized)
        B_quantizer.update_quantized(B_tensor, B_quantized)

        workspace = _empty_tensor()

        print(f"A quantized data: rowwise={A_quantized._rowwise_data is not None}, columnwise={A_quantized._columnwise_data is not None}")
        print(f"B quantized data: rowwise={B_quantized._rowwise_data is not None}, columnwise={B_quantized._columnwise_data is not None}")

        # Test cases for 1D1D kernel configurations
        test_cases = [
            {
                "name": "1D1D with float32 output + recipe + c_tensor",
                "out_dtype": torch.float32,
                "recipe": (1, 1, 128),
                "use_c_tensor": True,
                "use_rowwise_b": True
            },
            {
                "name": "1D1D with bfloat16 output + recipe + c_tensor",
                "out_dtype": torch.bfloat16,
                "recipe": (1, 1, 128),
                "use_c_tensor": True,
                "use_rowwise_b": True
            },
            {
                "name": "1D1D with float32 output + recipe + no c_tensor",
                "out_dtype": torch.float32,
                "recipe": (1, 1, 128),
                "use_c_tensor": False,
                "use_rowwise_b": True
            },
            {
                "name": "1D1D with bfloat16 output + recipe + no c_tensor",
                "out_dtype": torch.bfloat16,
                "recipe": (1, 1, 128),
                "use_c_tensor": False,
                "use_rowwise_b": True
            },
        ]

        all_tests_passed = True

        for test_case in test_cases:
            print(f"\n=== {test_case['name']} ===")

            try:
                # Get the appropriate B data based on test case
                if test_case['use_rowwise_b']:
                    if B_quantized._rowwise_data is None:
                        print("  ❌ Rowwise B data not available, skipping")
                        continue
                    B_data, B_scales = B_quantized._rowwise_data, B_quantized._rowwise_scale_inv
                    print(f"  Using rowwise B data: {B_data.shape}, scales: {B_scales.shape}")
                else:
                    B_data, B_scales = B_quantized._columnwise_data, B_quantized._columnwise_scale_inv
                    print(f"  Using columnwise B data: {B_data.shape}, scales: {B_scales.shape}")

                A_data, A_scales = A_quantized._rowwise_data, A_quantized._rowwise_scale_inv

                # Create output tensor
                output = torch.empty(M, N, device=device, dtype=test_case['out_dtype'])

                # Setup c_tensor
                c_tensor = output if test_case['use_c_tensor'] else None

                print(f"  Output dtype: {output.dtype}, c_tensor: {c_tensor is not None}, recipe: {test_case['recipe']}")

                # Call DeepGEMM directly with specific 1D1D parameters
                deep_gemm.fp8_gemm_nt(
                    (A_data, A_scales),
                    (B_data, B_scales),
                    output,
                    c=c_tensor,
                    recipe=test_case['recipe']
                )

                print(f"  ✅ Success! Output shape: {output.shape}, dtype: {output.dtype}")

                # Basic validation
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"  ❌ NaN or Inf detected in output")
                    all_tests_passed = False
                else:
                    print(f"  ✅ Output numerically valid")

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                all_tests_passed = False

        # Also test via the deepgemm_fp8_gemm interface with forced 1D1D parameters
        print(f"\n=== Testing via deepgemm_fp8_gemm interface ===")

        # I'll need to modify the deepgemm.py to support forcing 1D1D mode
        print("Note: Current deepgemm_fp8_gemm always uses 1D2D. Need to add 1D1D support.")

        # Summary
        print("\n" + "=" * 50)
        if all_tests_passed:
            print("🎉 1D1D kernel tests PASSED!")
            return True
        else:
            print("❌ Some 1D1D kernel tests FAILED!")
            return False

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_deepgemm_1d1d()

    if success:
        print("\n✅ DeepGEMM 1D1D test suite PASSED!")
        sys.exit(0)
    else:
        print("\n❌ DeepGEMM 1D1D test suite FAILED!")
        sys.exit(1)