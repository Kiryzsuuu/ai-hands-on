@echo off
echo 🚀 AI Learning Setup untuk Windows
echo =================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python tidak terdeteksi!
    echo.
    echo 💡 Silakan install Python dari: https://python.org
    echo    Pastikan centang "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo ✅ Python terdeteksi
echo.

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing AI libraries...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ⚠️ Error dengan requirements.txt, mencoba versi terbaru...
    echo 📦 Installing packages tanpa version lock (Python 3.13 compatible)...
    pip install scikit-learn numpy pandas matplotlib seaborn tensorflow pillow opencv-python tqdm jupyter
    
    if %errorlevel% neq 0 (
        echo.
        echo ❌ Masih ada error saat install libraries
        echo 💡 Solusi:
        echo    1. Pastikan internet connection stable
        echo    2. Run sebagai Administrator
        echo    3. Atau coba: conda install scikit-learn numpy pandas matplotlib
        echo.
        pause
        exit /b 1
    )
)

echo.
echo ✅ Setup selesai!
echo.
echo 🎯 Langkah selanjutnya:
echo    1. Jalankan: python 01_linear_regression.py
echo    2. Baca TUTORIAL.md
echo.
pause