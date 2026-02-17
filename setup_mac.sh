#!/bin/bash

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

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "📦 Installing pip3..."
    curl https://bootstrap.pypa.io/get-pip.py | python3
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install requirements
echo "📦 Installing AI libraries..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️ Error dengan requirements.txt, mencoba versi terbaru..."
    echo "📦 Installing packages tanpa version lock (Python 3.13 compatible)..."
    pip3 install scikit-learn numpy pandas matplotlib seaborn tensorflow pillow opencv-python tqdm jupyter
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Masih ada error saat install libraries"
        echo "💡 Solusi:"
        echo "   1. Coba dengan: sudo pip3 install [packages...]"
        echo "   2. Atau gunakan conda: conda install scikit-learn numpy pandas matplotlib"
        echo "   3. Check internet connection"
        echo ""
        exit 1
    fi
fi

# Make Python scripts executable
chmod +x *.py

echo ""
echo "✅ Setup selesai!"
echo ""
echo "🎯 Langkah selanjutnya:"
echo "   1. Jalankan: python3 01_linear_regression.py"
echo "   2. Baca TUTORIAL.md"
echo ""

# Optional: Run test
read -p "🧪 Ingin test installation? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 setup.py
fi