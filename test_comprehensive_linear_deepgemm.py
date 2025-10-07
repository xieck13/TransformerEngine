#!/usr/bin/env python3

"""Comprehensive LinearDeepGemm validation test"""

import torch
import warnings
from transformer_engine.pytorch.module.linear_deepgemm import LinearDeepGemm

def test_comprehensive_linear_deepgemm():
    """Test LinearDeepGemm comprehensively"""

    print("🧪 Comprehensive LinearDeepGemm Test Suite")
    print("=" * 50)

    device = 'cuda'

    # Test different sizes
    test_sizes = [
        (128, 256, 512),    # Our working case
        (256, 256, 256),    # Square matrices
        (64, 128, 256),     # Smaller case
    ]

    total_tests = 0
    passed_tests = 0

    for batch_size, in_features, out_features in test_sizes:
        print(f"\n--- Testing size: batch={batch_size}, in={in_features}, out={out_features} ---")

        try:
            # Create module
            linear = LinearDeepGemm(
                in_features=in_features,
                out_features=out_features,
                bias=True,
                accumulate_into_main_grad=False,
                device=device,
                dtype=torch.bfloat16
            )

            # Create input
            input_tensor = torch.randn(batch_size, in_features, device=device, dtype=torch.bfloat16, requires_grad=True)

            # Forward pass
            output = linear(input_tensor)
            print(f"  ✅ Forward: {input_tensor.shape} -> {output.shape}")

            # Backward pass
            loss = output.sum()
            loss.backward()

            print(f"  ✅ Backward: grad shapes - input: {input_tensor.grad.shape}, weight: {linear.weight.grad.shape}")

            # Check for NaN
            assert not torch.isnan(output).any(), "Output has NaN"
            assert not torch.isnan(input_tensor.grad).any(), "Input grad has NaN"
            assert not torch.isnan(linear.weight.grad).any(), "Weight grad has NaN"

            print(f"  ✅ No NaN values detected")
            print(f"  ✅ Test PASSED!")
            passed_tests += 1

        except Exception as e:
            print(f"  ❌ Test FAILED: {e}")

        total_tests += 1

    print(f"\n🎯 Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🚀 All comprehensive tests PASSED!")
        return True
    else:
        print("❌ Some comprehensive tests FAILED!")
        return False

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Skipping tests.")
        exit(1)

    success = test_comprehensive_linear_deepgemm()
    exit(0 if success else 1)