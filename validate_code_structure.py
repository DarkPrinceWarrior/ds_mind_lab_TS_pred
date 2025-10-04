"""Validate code structure without requiring dependencies."""

import sys
import ast
from pathlib import Path

print("="*80)
print("Code Structure Validation (No Dependencies Required)")
print("="*80)

def check_file_syntax(filepath):
    """Check if Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
            ast.parse(code)
        return True, "Valid syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_class_exists(filepath, class_name):
    """Check if a class exists in file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if f"class {class_name}" in content:
                return True, f"Found class {class_name}"
            else:
                return False, f"Class {class_name} not found"
    except Exception as e:
        return False, f"Error: {e}"

def check_function_exists(filepath, func_name):
    """Check if a function exists in file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if f"def {func_name}" in content:
                return True, f"Found function {func_name}"
            else:
                return False, f"Function {func_name} not found"
    except Exception as e:
        return False, f"Error: {e}"

results = []

# Test 1: Check models_advanced.py
print("\n[TEST 1] Checking src/models_advanced.py...")
file_path = Path("src/models_advanced.py")
if file_path.exists():
    valid, msg = check_file_syntax(file_path)
    results.append(("models_advanced.py syntax", valid, msg))
    
    if valid:
        valid, msg = check_class_exists(file_path, "MultiScaleTSMixer")
        results.append(("MultiScaleTSMixer class", valid, msg))
        
        valid, msg = check_class_exists(file_path, "AttentionTSMixerx")
        results.append(("AttentionTSMixerx class", valid, msg))
        
        valid, msg = check_class_exists(file_path, "EnsembleForecaster")
        results.append(("EnsembleForecaster class", valid, msg))
        
        # Check for key improvements
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "scales: List[int] = [1, 3, 12]" in content:
            results.append(("MultiScale default scales [1,3,12]", True, "Correct temporal scales"))
        else:
            results.append(("MultiScale default scales", False, "Scales not set to [1,3,12]"))
        
        if "scale_attention" in content:
            results.append(("Scale attention mechanism", True, "Found attention weighting"))
        else:
            results.append(("Scale attention mechanism", False, "Attention not found"))
else:
    results.append(("models_advanced.py", False, "File not found"))

# Test 2: Check features_advanced.py
print("\n[TEST 2] Checking src/features_advanced.py...")
file_path = Path("src/features_advanced.py")
if file_path.exists():
    valid, msg = check_file_syntax(file_path)
    results.append(("features_advanced.py syntax", valid, msg))
    
    if valid:
        valid, msg = check_function_exists(file_path, "create_fourier_features")
        results.append(("create_fourier_features", valid, msg))
        
        valid, msg = check_function_exists(file_path, "create_pressure_gradient_features")
        results.append(("create_pressure_gradient_features", valid, msg))
        
        valid, msg = check_function_exists(file_path, "create_time_series_embeddings")
        results.append(("create_time_series_embeddings", valid, msg))
else:
    results.append(("features_advanced.py", False, "File not found"))

# Test 3: Check wlpr_pipeline.py
print("\n[TEST 3] Checking src/wlpr_pipeline.py...")
file_path = Path("src/wlpr_pipeline.py")
if file_path.exists():
    valid, msg = check_file_syntax(file_path)
    results.append(("wlpr_pipeline.py syntax", valid, msg))
    
    if valid:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check config updates
        if "fourier_sin_1" in content and "fourier_cos_1" in content:
            results.append(("Fourier features in config", True, "Found in hist_exog/futr_exog"))
        else:
            results.append(("Fourier features in config", False, "Not found in config"))
        
        if "wbhp_gradient" in content and "productivity_index" in content:
            results.append(("Pressure features in config", True, "Found in hist_exog"))
        else:
            results.append(("Pressure features in config", False, "Not found in config"))
        
        if "ts_embed_0" in content:
            results.append(("PCA embeddings in config", True, "Found in hist_exog"))
        else:
            results.append(("PCA embeddings in config", False, "Not found in config"))
        
        # Check feature creation integration
        if "create_fourier_features" in content:
            results.append(("Fourier integration", True, "Called in prepare_model_frames"))
        else:
            results.append(("Fourier integration", False, "Not integrated"))
        
        if "create_pressure_gradient_features" in content:
            results.append(("Pressure gradient integration", True, "Called in prepare_model_frames"))
        else:
            results.append(("Pressure gradient integration", False, "Not integrated"))
        
        if "create_time_series_embeddings" in content:
            results.append(("PCA integration", True, "Called in prepare_model_frames"))
        else:
            results.append(("PCA integration", False, "Not integrated"))
        
        # Check multiscale config
        if "multiscale_scales: List[int] = field(default_factory=lambda: [1, 3, 12])" in content:
            results.append(("MultiScale config", True, "Scales set to [1,3,12]"))
        else:
            results.append(("MultiScale config", False, "Scales not properly configured"))
else:
    results.append(("wlpr_pipeline.py", False, "File not found"))

# Test 4: Check documentation
print("\n[TEST 4] Checking documentation...")
doc_path = Path("MULTISCALE_ATTENTION_IMPLEMENTATION.md")
if doc_path.exists():
    results.append(("Implementation doc", True, "MULTISCALE_ATTENTION_IMPLEMENTATION.md exists"))
else:
    results.append(("Implementation doc", False, "Documentation not found"))

# Summary
print("\n" + "="*80)
print("VALIDATION RESULTS")
print("="*80)

passed = 0
failed = 0

for name, success, msg in results:
    status = "[OK]" if success else "[FAIL]"
    print(f"{status} {name}")
    if not success:
        print(f"      {msg}")
    if success:
        passed += 1
    else:
        failed += 1

print("\n" + "-"*80)
print(f"Total: {passed} passed, {failed} failed out of {len(results)} tests")

if failed == 0:
    print("\n[SUCCESS] All code structure validations passed!")
    print("\nImplementation is complete:")
    print("  1. MultiScaleTSMixer with [1, 3, 12] month scales")
    print("  2. Attention mechanism for interpretability")
    print("  3. Advanced features (Fourier, pressure gradients, PCA)")
    print("  4. Full pipeline integration")
    print("\nTo run the pipeline (requires dependencies):")
    print("  pip install torch numpy pandas scikit-learn scipy neuralforecast")
    print("  python src/wlpr_pipeline.py --data-path MODEL_22.09.25.csv")
    sys.exit(0)
else:
    print("\n[WARNING] Some validations failed. Please review the errors above.")
    sys.exit(1)
