#!/bin/bash

set -u

echo "🚀 AI Learning Setup for Mac/Linux"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 tidak terdeteksi!"
    echo ""
    echo "💡 Install Python3:"
    echo "   Mac: brew install python3"
    echo "   Ubuntu: sudo apt install python3 python3-pip"
    echo ""
    exit 1
fi

echo "✅ Python3 terdeteksi"
python3 --version
echo ""

# Create & activate venv (recommended)
if [ ! -d ".venv" ]; then
    echo "🐍 Membuat virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "🔌 Mengaktifkan virtual environment (.venv)..."
# shellcheck disable=SC1091
source .venv/bin/activate

# Determine Python version (major/minor)
PY_MAJOR=$(python -c 'import sys; print(sys.version_info[0])')
PY_MINOR=$(python -c 'import sys; print(sys.version_info[1])')
TF_SUPPORTED=1
if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 13 ]; then
    TF_SUPPORTED=0
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "📦 Installing pip3..."
    curl https://bootstrap.pypa.io/get-pip.py | python3
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📦 Installing AI libraries..."
python -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️ Error dengan requirements.txt, mencoba versi terbaru..."
    echo "📦 Installing packages tanpa version lock (core packages)..."
    python -m pip install scikit-learn numpy pandas matplotlib seaborn pillow opencv-python tqdm jupyter python-pptx

    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Masih ada error saat install libraries (core packages)"
        echo "💡 Solusi:"
        echo "   1. Pastikan virtual environment aktif: source .venv/bin/activate"
        echo "   2. Update pip: python -m pip install --upgrade pip"
        echo "   3. Atau gunakan conda/miniforge"
        echo ""
        exit 1
    fi

    if [ "$TF_SUPPORTED" -eq 1 ]; then
        echo ""
        echo "🧠 Installing TensorFlow (untuk Program 3/4)..."
        python -m pip install tensorflow
    else
        echo ""
        echo "⚠️ TensorFlow di-skip karena Python kamu $PY_MAJOR.$PY_MINOR"
        echo "   TensorFlow biasanya belum tersedia untuk Python 3.13/3.14."
        echo "   Untuk menjalankan Program 3/4, gunakan Python 3.10–3.12 (mis. via pyenv/conda),"
        echo "   lalu install: pip install tensorflow"
    fi
fi

# Make Python scripts executable
chmod +x *.py

echo ""
echo "✅ Setup selesai!"
echo ""
echo "🎯 Langkah selanjutnya:"
echo "   1. Jalankan: python 01_linear_regression.py"
echo "   2. Baca TUTORIAL.md"
if [ "$TF_SUPPORTED" -eq 0 ]; then
    echo "   3. Catatan: Program 3/4 butuh TensorFlow (Python 3.10–3.12)."
fi
echo ""

# Optional: Run test
read -p "🧪 Ingin test installation? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python setup.py
fi