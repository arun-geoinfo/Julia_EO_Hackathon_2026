"""
verify_onnx.py
Verify the exported ONNX model works correctly.
Run this BEFORE using the model in Julia.
"""
import onnxruntime as ort
import numpy as np
import sys
# EVA-02 normalization constants
EVA02_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 3, 1, 1)
EVA02_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 3, 1, 1)
def verify_onnx_model(onnx_path="transformer_model.onnx"):
    """Verify the ONNX model meets all requirements."""
   
    print("=" * 60)
    print("🔍 ONNX MODEL VERIFICATION")
    print("=" * 60)
   
    # 1. Load the ONNX model
    print("\n1. Loading ONNX model...")
    try:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        print(f" ✅ Model loaded: {onnx_path}")
    except Exception as e:
        print(f" ❌ Failed to load model: {e}")
        return False
   
    # 2. Check input/output details
    print("\n2. Model metadata:")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
   
    print(f" Number of inputs: {len(inputs)}")
    for i, inp in enumerate(inputs):
        print(f" Input {i}: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
   
    print(f" Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f" Output {i}: {out.name}, Shape: {out.shape}, Type: {out.type}")
   
    # 3. Test with various inputs
    print("\n3. Testing with various inputs...")
   
    test_cases = [
        ("Gray image (0.5)", np.ones((1, 3, 448, 448), dtype=np.float32) * 0.5),
        ("Black image", np.zeros((1, 3, 448, 448), dtype=np.float32)),
        ("White image", np.ones((1, 3, 448, 448), dtype=np.float32)),
        ("Random image", np.random.randn(1, 3, 448, 448).astype(np.float32)),
    ]
   
    all_passed = True
    for name, input_data in test_cases:
        try:
            # Normalize input (pixels in [0, 1] range)
            # The model expects normalized input in [0, 1] range
            input_normalized = np.clip(input_data, 0, 1)
           
            # Run inference
            outputs = session.run(None, {'input': input_normalized})
            output_data = outputs[0]
           
            # Check output properties
            output_sum = output_data.sum()
            output_min = output_data.min()
            output_max = output_data.max()
            output_shape = output_data.shape
           
            # CRITICAL CHECKS
            checks = [
                (abs(output_sum - 1.0) < 0.01, f"Sum ≈ 1.0 (actual: {output_sum:.6f})"),
                (output_min >= 0, f"All values ≥ 0 (min: {output_min:.6f})"),
                (output_max <= 1.0, f"All values ≤ 1.0 (max: {output_max:.6f})"),
                (output_shape == (1, 1000), f"Shape = (1, 1000) (actual: {output_shape})"),
            ]
           
            print(f"\n Test: {name}")
            passed = True
            for check_passed, check_msg in checks:
                status = "✅" if check_passed else "❌"
                print(f" {status} {check_msg}")
                if not check_passed:
                    passed = False
                    all_passed = False
           
            if passed:
                print(f" ✅ All checks passed")
            else:
                print(f" ❌ Some checks failed")
               
        except Exception as e:
            print(f" ❌ Test '{name}' failed: {e}")
            all_passed = False
   
    # 4. Test batch inference
    print("\n4. Testing batch inference...")
    try:
        batch_size = 4
        batch_input = np.random.rand(batch_size, 3, 448, 448).astype(np.float32)
       
        outputs = session.run(None, {'input': batch_input})
        output_data = outputs[0]
       
        print(f" Input shape: {batch_input.shape}")
        print(f" Output shape: {output_data.shape}")
       
        if output_data.shape == (batch_size, 1000):
            print(f" ✅ Batch output shape correct")
           
            # Check each sample in batch
            for i in range(batch_size):
                sample_sum = output_data[i].sum()
                if abs(sample_sum - 1.0) < 0.01:
                    print(f" Sample {i}: ✅ Sum = {sample_sum:.6f}")
                else:
                    print(f" Sample {i}: ❌ Sum = {sample_sum:.6f}")
                    all_passed = False
        else:
            print(f" ❌ Batch output shape incorrect")
            all_passed = False
           
    except Exception as e:
        print(f" ❌ Batch test failed: {e}")
        all_passed = False
   
    # 5. Final verification
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION:")
    print("=" * 60)
   
    if all_passed:
        print("🎉 ALL TESTS PASSED! Model is ready for Julia.")
        print("\nCRITICAL INFORMATION FOR JULIA:")
        print(" 1. Input name: 'input'")
        print(" 2. Output name: 'probabilities'")
        print(" 3. Input shape: (batch, 3, 448, 448)")
        print(" 4. Output shape: (batch, 1000)")
        print(" 5. Input range: [0, 1] (divide by 255)")
        print(" 6. Normalization: EVA-02 (built into model)")
        print(" 7. Output: Probabilities (sum ≈ 1.0)")
        return True
    else:
        print("❌ SOME TESTS FAILED! Do not use in Julia.")
        return False
def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else "transformer_model.onnx"
   
    print(f"Verifying ONNX model: {onnx_path}")
    print("This verifies the model meets all requirements for Julia feature extraction.\n")
   
    success = verify_onnx_model(onnx_path)
   
    if success:
        print("\n✅ Model verification successful!")
        print("You can now use the model in Julia.")
    else:
        print("\n❌ Model verification failed!")
        print("Do not proceed to Julia until issues are fixed.")
   
    sys.exit(0 if success else 1)
if __name__ == "__main__":
    main()
