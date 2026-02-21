#!/usr/bin/env python3
"""
🚀 SETUP SCRIPT - AI Learning untuk Pemula
Script otomatis untuk setup environment dan test installation
"""

import sys
import subprocess
import importlib
import os

def print_header():
    print("🤖" + "=" * 60 + "🤖")
    print("     SETUP AI LEARNING ENVIRONMENT")
    print("     untuk Windows & Mac")
    print("🤖" + "=" * 60 + "🤖")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    if sys.version_info < (3, 7):
        print(f"❌ Python {sys.version_info[0]}.{sys.version_info[1]} detected")
        print("   AI Learning requires Python 3.7+")
        print("   Please upgrade Python from python.org")
        return False
    else:
        print(f"✅ Python {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]} detected - OK!")
        return True

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"   Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_packages():
    """Check and install required packages"""
    print("\n📦 Checking and installing required packages...")

    # List of required packages (core)
    packages = [
        "numpy", "pandas", "matplotlib", "seaborn",
        "scikit-learn", "Pillow", "opencv-python",
        "tqdm", "jupyter", "python-pptx"
    ]

    # TensorFlow is optional because it often lags behind newest Python versions
    tf_supported = sys.version_info < (3, 13)
    if tf_supported:
        packages.insert(5, "tensorflow")
    else:
        print(f"\n⚠️  TensorFlow di-skip (Python {sys.version_info[0]}.{sys.version_info[1]})")
        print("   Program 3/4 membutuhkan TensorFlow; gunakan Python 3.10–3.12 untuk itu.")
    
    installed = []
    failed = []
    
    for package in packages:
        try:
            # Try to import the package
            if package == "opencv-python":
                importlib.import_module("cv2")
            elif package == "Pillow":
                importlib.import_module("PIL")
            else:
                importlib.import_module(package.replace("-", "_"))
            
            print(f"   ✅ {package} already installed")
            installed.append(package)
            
        except ImportError:
            print(f"   📥 {package} not found, installing...")
            if install_package(package):
                print(f"   ✅ {package} installed successfully")
                installed.append(package)
            else:
                print(f"   ❌ Failed to install {package}")
                failed.append(package)
    
    print(f"\n📊 Installation Summary:")
    print(f"   ✅ Installed: {len(installed)}/{len(packages)} packages")
    
    if failed:
        print(f"   ❌ Failed: {failed}")
        print(f"   💡 Try: pip install --upgrade pip")
        print(f"   💡 Then: pip install {' '.join(failed)}")
        return False
    
    return True

def test_basic_imports():
    """Test if we can import basic libraries"""
    print("\n🧪 Testing basic AI library imports...")

    tests = [
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("scikit-learn", "from sklearn.linear_model import LinearRegression"),
    ]

    if sys.version_info < (3, 13):
        tests.append(("tensorflow", "import tensorflow as tf"))
    
    all_pass = True
    
    for name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"   ✅ {name} - OK")
        except Exception as e:
            print(f"   ❌ {name} - FAILED: {str(e)[:50]}...")
            all_pass = False
    
    return all_pass

def create_test_script():
    """Create a simple test script"""
    print("\n📝 Creating test script...")
    
    test_script = '''
# Quick AI test script
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("🤖 AI Libraries Test - SUCCESS!")
print(f"NumPy version: {np.__version__}")

# Generate simple data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make prediction
prediction = model.predict([[6]])[0]
print(f"Prediction for input 6: {prediction}")

# Simple plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label='Model')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression Test')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('ai_test_result.png', dpi=100, bbox_inches='tight')
print("✅ Test plot saved as 'ai_test_result.png'")
print("🎉 All systems ready for AI learning!")
'''
    
    with open('test_ai_setup.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("   ✅ test_ai_setup.py created")
    
    # Run the test
    try:
        print("   🧪 Running test script...")
        exec(test_script)
        return True
    except Exception as e:
        print(f"   ❌ Test failed: {str(e)}")
        return False

def show_next_steps():
    """Show what user should do next"""
    print("\n🎯 NEXT STEPS:")
    print()
    print("1️⃣  Start with Linear Regression:")
    print("    python 01_linear_regression.py")
    print()
    print("2️⃣  Read the tutorial:")
    print("    Open TUTORIAL.md in any text editor")
    print()
    print("3️⃣  Follow the learning path:")
    print("    01 → 02 → 03 → 04 (berturut-turut)")
    print()
    print("4️⃣  Join komunitas:")
    print("    - Indonesia Belajar (Discord)")
    print("    - r/MachineLearning (Reddit)")
    print("    - Kaggle Learn (Online)")
    print()

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup cannot continue. Please upgrade Python first.")
        return
    
    # Check and install packages
    if not check_and_install_packages():
        print("\n❌ Some packages failed to install. Please check your internet connection.")
        print("💡 Try running: pip install --upgrade pip")
        return
    
    # Test imports
    if not test_basic_imports():
        print("\n❌ Some libraries are not working properly.")
        print("💡 Try reinstalling problematic packages.")
        return
    
    # Create and run test
    if not create_test_script():
        print("\n⚠️  Test script had issues, but basic setup should work.")
    
    print("\n" + "🎉" * 20)
    print("    SETUP COMPLETED SUCCESSFULLY!")
    print("🎉" * 20)
    
    show_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please check your Python installation and try again.")