# 📚 BUKU PANDUAN LENGKAP: Belajar AI untuk Pemula
## *Dari Nol Sampai Mahir - Panduan Praktis dengan Penjelasan Mendalam*

**Versi 1.0 - Februari 2026**  
**Penulis: AI Learning Indonesia**  
**Target: Pemula yang ingin menguasai AI fundamental**

---

## 📑 DAFTAR ISI

### BAGIAN I: FONDASI AI
1. [Pengenalan Artificial Intelligence](#bagian-1-pengenalan-ai)
2. [Setup Environment & Troubleshooting](#bagian-2-setup-environment)
3. [Konsep Fundamental Machine Learning](#bagian-3-konsep-fundamental)

### BAGIAN II: PRAKTEK HANDS-ON  
4. [Modul 1: Linear Regression - Fondasi ML](#modul-1-linear-regression)
5. [Modul 2: Classification - Decision Making](#modul-2-classification)
6. [Modul 3: Neural Network - Deep Learning](#modul-3-neural-network)
7. [Modul 4: Computer Vision - AI yang Melihat](#modul-4-computer-vision)

### BAGIAN III: PENGUASAAN
8. [Mengapa Algoritma Bekerja Seperti Itu?](#bagian-8-mengapa-algoritma)
9. [Troubleshooting & Problem Solving](#bagian-9-troubleshooting)
10. [Next Level: Project Ideas & Career Path](#bagian-10-next-level)

---

## PENGANTAR

> *"Setiap ahli dulunya adalah pemula. Yang membedakan adalah konsistensi belajar dan praktik."*

### 🎯 Mengapa Buku Ini Ada?

Artificial Intelligence (AI) sering terlihat seperti **magic** ✨. Padahal, di baliknya hanyalah **matematika, statistik, dan algoritma** yang bisa dipelajari siapa saja. 

Buku ini dirancang untuk:
- **Pemula total** yang belum pernah coding AI
- **Mahasiswa** yang ingin memahami AI secara mendalam
- **Professional** yang ingin transition ke AI field
- **Siapa saja** yang penasaran bagaimana AI bekerja

### 🔥 Keunikan Buku Ini

✅ **Bahasa Indonesia** - Semua konsep dijelaskan dalam bahasa yang mudah dipahami  
✅ **Praktik Langsung** - Setiap teori langsung dipraktikkan  
✅ **Penjelasan Mendalam** - Tidak hanya "cara", tapi "mengapa"  
✅ **Troubleshooting Guide** - Solusi untuk semua masalah umum  
✅ **Real Code** - Program yang bisa dijalankan dan dimodifikasi  

### 📖 Cara Menggunakan Buku Ini

1. **Baca secara berurutan** - Jangan skip bab!
2. **Praktik sambil baca** - Run code untuk setiap konsep
3. **Eksperimen** - Ubah parameter, lihat apa yang terjadi
4. **Catat insight** - Tulis pemahaman Anda sendiri
5. **Ulangi yang sulit** - AI butuh repetition untuk dipahami

---

## BAGIAN 1: PENGENALAN AI

### 1.1 Apa itu Artificial Intelligence?

**Definisi Sederhana:**
AI adalah teknologi yang membuat komputer bisa "berpikir" dan membuat keputusan seperti manusia.

**Analogi Mudah:**
Bayangkan AI seperti **murid yang sangat rajin**:
- Diberi contoh berkali-kali (training data)
- Belajar mengenali pola (algorithm)  
- Bisa menjawab pertanyaan baru (prediction)
- Semakin banyak belajar, semakin pintar (model improvement)

### 1.2 Hierarki AI

```
🧠 ARTIFICIAL INTELLIGENCE (AI)
├── 🤖 MACHINE LEARNING (ML)
│   ├── 👨‍🏫 Supervised Learning
│   │   ├── 📊 Classification (kategori: kucing/anjing)
│   │   └── 📈 Regression (angka: harga rumah)
│   ├── 🔍 Unsupervised Learning
│   │   ├── 🎯 Clustering (kelompokkan data)
│   │   └── 📏 Dimensionality Reduction
│   └── 🎮 Reinforcement Learning (belajar dari reward)
└── 🧠 DEEP LEARNING
    ├── 🕸️ Neural Networks (imitasi otak)
    ├── 👁️ Computer Vision (penglihatan)
    ├── 🗣️ Natural Language Processing (bahasa)
    └── 🎨 Generative AI (ChatGPT, DALL-E)
```

### 1.3 Mengapa AI Penting di 2026?

**Statistik Dunia:**
- 🏢 **50%** perusahaan menggunakan AI untuk operasional
- 💼 **85%** job baru membutuhkan AI literacy
- 🚀 **$126 miliar** investasi AI global per tahun
- 🎯 **3x** lebih produktif dengan AI tools

**Di Indonesia:**
- 🏦 Banking: Fraud detection, credit scoring
- 🛒 E-commerce: Recommendation systems
- 🚗 Transportation: Route optimization  
- 🏥 Healthcare: Medical diagnosis assistance

---

## BAGIAN 2: SETUP ENVIRONMENT

### 2.1 Mengapa Setup Penting?

**Analogi:**
Setup environment seperti **menyiapkan dapur** sebelum masak. Kalau alatnya tidak lengkap atau rusak, hasilnya pasti tidak optimal.

### 2.2 Requirements Checklist

#### ✅ Hardware Minimum:
- **RAM**: 8GB (recommended 16GB)
- **Storage**: 5GB free space
- **Processor**: Intel i5/AMD Ryzen 5 atau yang lebih baik
- **Internet**: Stabil untuk download dataset

#### ✅ Software Requirements:
- **Python**: 3.10+ (recommended 3.11 atau 3.12)
- **Code Editor**: VS Code, PyCharm, atau Jupyter
- **Terminal**: Command Prompt/PowerShell (Windows) atau Terminal (Mac/Linux)

### 2.3 Step-by-Step Installation

#### **Windows Setup:**

```bash
# 1. Download Python dari python.org
# ✅ PASTIKAN centang "Add Python to PATH"

# 2. Buka Command Prompt/PowerShell di folder project
# 3. Buat virtual environment
python -m venv ai-learning-env

# 4. Aktivasi environment  
ai-learning-env\Scripts\activate

# 5. Update pip
python -m pip install --upgrade pip

# 6. Install packages
pip install scikit-learn numpy pandas matplotlib seaborn pillow opencv-python tqdm jupyter python-pptx

# (Opsional) TensorFlow untuk Program 3/4
# Biasanya paling stabil di Python 3.10–3.12:
# pip install tensorflow
```

#### **Mac Setup:**

```bash
# 1. Install Homebrew (jika belum ada)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python
brew install python3

# 3. Buat virtual environment
python3 -m venv ai-learning-env

# 4. Aktivasi environment
source ai-learning-env/bin/activate

# 5. Install packages  
pip3 install scikit-learn numpy pandas matplotlib seaborn pillow opencv-python tqdm jupyter python-pptx

# (Opsional) TensorFlow untuk Program 3/4
# Biasanya paling stabil di Python 3.10–3.12:
# pip3 install tensorflow
```

### 2.4 Troubleshooting Common Issues

#### ❌ Problem: "ModuleNotFoundError"
```
Traceback (most recent call last):
  File "program.py", line 1, in <module>
    import numpy
ModuleNotFoundError: No module named 'numpy'
```

**🔧 Solution:**
```bash
# Check apakah virtual environment aktif
# Lihat prompt: (ai-learning-env) atau tidak?

# Jika tidak aktif, aktivasi dulu:
# Windows:
ai-learning-env\Scripts\activate

# Mac/Linux:  
source ai-learning-env/bin/activate

# Install package yang missing:
pip install numpy
```

#### ❌ Problem: "Permission Denied" (Mac/Linux)
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**🔧 Solution:**
```bash
# Jangan gunakan sudo! Gunakan virtual environment:
python3 -m venv ai-learning-env
source ai-learning-env/bin/activate
pip3 install [package-name]
```

#### ❌ Problem: "Microsoft Visual C++ 14.0 required" (Windows)
```
error: Microsoft Visual C++ 14.0 is required
```

**🔧 Solution:**
```bash
# Download dan install Microsoft Build Tools:
# https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019

# Atau gunakan conda sebagai alternative:
conda install scikit-learn numpy pandas matplotlib
```

#### ❌ Problem: Matplotlib grafik tidak muncul
```
Backend TkAgg is interactive backend. Turning interactive mode on.
# Tapi tidak ada jendela yang muncul
```

**🔧 Solution:**
```python
# Tambahkan di awal program:
import matplotlib
matplotlib.use('TkAgg')  # Atau 'Qt5Agg'
import matplotlib.pyplot as plt

# Jika masih bermasalah, gunakan backend non-interactive:
matplotlib.use('Agg')
plt.savefig('output.png')  # Save ke file instead
```

---

## BAGIAN 3: KONSEP FUNDAMENTAL

### 3.1 Machine Learning vs Programming Tradisional

#### **Programming Tradisional:**
```
Input + Algorithm = Output

Contoh:
Suhu Celsius + Rumus Konversi = Suhu Fahrenheit
```

#### **Machine Learning:**
```
Input + Output = Algorithm

Contoh:
Data Rumah + Harga Rumah = Model Prediksi
```

### 3.2 Supervised vs Unsupervised Learning

#### **Supervised Learning (Dengan Guru)**

**Karakteristik:**
- Ada **target/label** yang ingin diprediksi
- Model belajar dari "contoh soal + jawaban"
- Bisaeval performa dengan membandingkan prediksi vs jawaban sebenarnya

**Jenis:**
1. **Classification** - Prediksi kategori
   - Email: Spam atau Bukan Spam
   - Gambar: Kucing, Anjing, atau Burung
   - Medis: Positif atau Negatif

2. **Regression** - Prediksi angka
   - Harga rumah berdasarkan fitur
   - Suhu besok berdasarkan data cuaca
   - Gaji berdasarkan pengalaman

#### **Unsupervised Learning (Tanpa Guru)**

**Karakteristik:**  
- Tidak ada target/label
- Model mencari pola tersembunyi dalam data
- Lebih sulit evaluate karena tidak ada "jawaban benar"

**Jenis:**
1. **Clustering** - Mengelompokkan data
   - Customer segmentation
   - Gene sequencing  
   - Social network analysis

2. **Dimensionality Reduction** - Simplifikasi data
   - Data compression
   - Visualization
   - Feature extraction

### 3.3 Overfitting vs Underfitting

**Analogi Restaurant:**

#### **Overfitting = Chef yang Terlalu Hafal**
- Chef hafal exact pesanan 100 customer terakhir
- Kalau customer baru pesan "nasi goreng agak pedas" → bingung
- Model terlalu spesifik ke training data
- **Solusi:** Validation set, regularization, more data

#### **Underfitting = Chef yang Malas Belajar** 
- Chef cm bisa masak 1 menu: nasi putih
- Semua pesanan dikasih nasi putih
- Model terlalu simple, tidak capture kompleksitas
- **Solusi:** Model lebih complex, more features, more training

#### **Good Fit = Chef Professional**
- Paham prinsip dasar cooking
- Bisa adaptasi ke permintaan baru
- Balance antara generalization vs specialization

### 3.4 Training vs Validation vs Test Set

**Analogi Sekolah:**

#### **Training Set (70%) = Buku Pelajaran**
- Data untuk model "belajar"
- Model lihat ini berkali-kali
- Optimasi parameter berdasarkan data ini

#### **Validation Set (15%) = Quiz Harian**  
- Data untuk tune hyperparameter
- Model TIDAK belajar dari ini
- Digunakan untuk pilih model terbaik

#### **Test Set (15%) = Ujian Akhir**
- Data untuk evaluasi final
- Model sama sekali TIDAK pernah lihat ini
- Simulasi performa di dunia nyata

```python
# Contoh splitting:
from sklearn.model_selection import train_test_split

# Split 1: Training vs (Validation + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Split 2: Validation vs Test  
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Hasil:
# X_train: 70% data untuk training
# X_val: 15% data untuk validation  
# X_test: 15% data untuk testing
```

### 3.5 Evaluation Metrics 

#### **Untuk Regression:**

1. **Mean Absolute Error (MAE)**
```
MAE = (|actual₁ - prediction₁| + |actual₂ - prediction₂| + ...) / n

Interpretasi: Rata-rata kesalahan dalam unit yang sama
Contoh: MAE = 50,000 → rata-rata salah prediksi 50 juta rupiah
```

2. **Root Mean Squared Error (RMSE)**  
```
RMSE = √((actual₁ - prediction₁)² + (actual₂ - prediction₂)² + ...) / n)

Interpretasi: Lebih menghukum kesalahan besar
Contoh: RMSE = 75,000 → error lebih bervariasi
```

3. **R-squared (R²)**
```
R² = 1 - (Sum of Squared Errors / Total Sum of Squares)

Interpretasi: Persentase variance yang dijelaskan model
Contoh: R² = 0.85 → model menjelaskan 85% variasi data
```

#### **Untuk Classification:**

1. **Accuracy**
```
Accuracy = (Correct Predictions) / (Total Predictions)

Contoh: 85/100 = 85% accuracy
```

2. **Precision**
```
Precision = True Positives / (True Positives + False Positives)

Interpretasi: Dari yang diprediksi positif, berapa yang benar?
Contoh: Dari 100 email yang diprediksi spam, 90 benar spam = 90% precision
```

3. **Recall (Sensitivity)**
```
Recall = True Positives / (True Positives + False Negatives)  

Interpretasi: Dari yang sebenarnya positif, berapa yang berhasil dideteksi?
Contoh: Dari 100 email spam yang ada, 85 terdeteksi = 85% recall
```

4. **F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Interpretasi: Harmonic mean dari Precision dan Recall
Berguna saat data imbalanced
```

**Confusion Matrix Visualization:**
```
                 PREDICTION
                 Pos    Neg
ACTUAL    Pos    TP     FN
          Neg    FP     TN

TP = True Positive (Benar prediksi positif)
TN = True Negative (Benar prediksi negatif)  
FP = False Positive (Salah prediksi positif) 
FN = False Negative (Salah prediksi negatif)
```

---

## MODUL 1: LINEAR REGRESSION

### 1.1 Konsep Dasar

**Linear Regression** adalah algoritma untuk memprediksi nilai **kontinyu** berdasarkan hubungan **linear** antar variabel.

**Rumus Matematika:**
```
y = mx + b

Dimana:  
y = Output yang ingin diprediksi
x = Input/fitur  
m = Slope/kemiringan (weight)
b = Intercept (bias)
```

**Contoh Real-world:**
- Prediksi harga rumah berdasarkan ukuran
- Prediksi gaji berdasarkan tahun pengalaman  
- Prediksi suhu berdasarkan ketinggian

### 1.2 Mengapa Linear Regression Penting?

1. **Foundation**: Basis untuk algoritma ML yang lebih kompleks
2. **Interpretable**: Mudah dipahami dan dijelaskan
3. **Fast**: Training dan prediction sangat cepat
4. **Baseline**: Standard untuk comparison dengan model lain

### 1.3 Hands-On Practice

#### **Step 1: Import Libraries**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Setting untuk reproducible results
np.random.seed(42)
```

**💡 Penjelasan:**
- `numpy`: Operasi matematika dan array
- `pandas`: Manipulasi data (seperti Excel di Python)
- `matplotlib`: Membuat grafik dan visualisasi
- `sklearn`: Library machine learning terlengkap

#### **Step 2: Generate Sample Data**
```python
# Buat data contoh: ukuran rumah vs harga
n_samples = 100
ukuran_rumah = np.random.randint(50, 301, n_samples)  # 50-300 m²

# Formula: harga = ukuran × 5 juta + noise
harga_rumah = ukuran_rumah * 5 + np.random.randint(-500, 501, n_samples)

# Pastikan harga tidak negatif  
harga_rumah = np.maximum(harga_rumah, 100)

print(f"Dataset dibuat: {n_samples} rumah")
print(f"Ukuran minimum: {ukuran_rumah.min()}m²")  
print(f"Ukuran maksimum: {ukuran_rumah.max()}m²")
print(f"Harga minimum: {harga_rumah.min()} juta")
print(f"Harga maksimum: {harga_rumah.max()} juta")
```

**💡 Mengapa Generate Data?**
- Real dataset seringkali messy dan kompleks untuk pemula
- Generated data memungkinkan kita kontrol exact relationship
- Bisa experiment dengan different patterns dan noise levels

#### **Step 3: Data Visualization**
```python
plt.figure(figsize=(10, 6))
plt.scatter(ukuran_rumah, harga_rumah, alpha=0.6, color='blue')
plt.xlabel('Ukuran Rumah (m²)')
plt.ylabel('Harga Rumah (Juta Rupiah)')  
plt.title('Hubungan Ukuran vs Harga Rumah')
plt.grid(True, alpha=0.3)
plt.show()
```

**💡 Insight yang Diharapkan:**
- Scatter plot menunjukkan **positive correlation**
- Pola **roughly linear** terlihat jelas
- Ada beberapa **outliers** (normal dalam real data)
- **Variance** meningkat seiring ukuran (heteroscedasticity)

#### **Step 4: Prepare Data for ML**
```python
# Reshape untuk scikit-learn (perlu 2D array)
X = ukuran_rumah.reshape(-1, 1)  # Features
y = harga_rumah                   # Target

# Split data: 80% training, 20% testing  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```

**💡 Kenapa Reshape?**
- Scikit-learn expect 2D arrays: [samples, features]
- Ukuran rumah shape: (100,) → perlu jadi (100, 1)
- Ini pattern standard untuk semua ML algorithms

#### **Step 5: Train the Model**
```python
# Buat dan train model
model = LinearRegression()
model.fit(X_train, y_train)

# Ekstrak parameters yang dipelajari
slope = model.coef_[0] 
intercept = model.intercept_

print(f"Model berhasil dilatih!")
print(f"Rumus: Harga = {slope:.2f} × Ukuran + {intercept:.2f}")
print(f"Interpretasi: Setiap 1m² menambah {slope:.0f} juta rupiah")
```

**💡 What Happened During Training:**
1. Algorithm mencari **best line** yang meminimalkan error
2. Menggunakan **Ordinary Least Squares** method  
3. Optimization: minimize Sum of Squared Errors
4. Result: optimal values untuk slope dan intercept

#### **Step 6: Make Predictions**
```python
# Prediksi pada test set
y_pred = model.predict(X_test)

# Test beberapa contoh
contoh_ukuran = [100, 150, 200, 250]
for ukuran in contoh_ukuran:
    prediksi = model.predict([[ukuran]])[0]
    print(f"Rumah {ukuran}m² → Prediksi harga: Rp {prediksi:.0f} juta")
```

#### **Step 7: Evaluate Performance**
```python  
# Hitung metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Performa Model:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.3f}")

# Interpretasi R²
if r2 > 0.8:
    print("🎉 Model sangat baik!")
elif r2 > 0.6:
    print("👍 Model cukup baik") 
elif r2 > 0.4:
    print("📈 Model OK, masih bisa diperbaiki")
else:
    print("📚 Model perlu improvement")
```

#### **Step 8: Visualize Results**
```python
plt.figure(figsize=(12, 5))

# Subplot 1: Predictions vs Actual
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')  
plt.title('Predictions vs Actual')

# Subplot 2: Regression Line
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.7, label='Actual')
plt.scatter(X_test, y_pred, alpha=0.7, color='red', label='Predicted')

# Plot regression line
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = model.predict(X_range)
plt.plot(X_range, y_range_pred, 'green', linewidth=2, label='Regression Line')

plt.xlabel('Ukuran Rumah (m²)')
plt.ylabel('Harga (Juta)')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 1.4 Mengapa Model Ini Bekerja?

#### **Matematika di Balik Linear Regression:**

1. **Objective Function:**
   ```
   Minimize: Σ(actual_i - predicted_i)²
   
   Where predicted_i = mx_i + b
   ```

2. **Optimization Process:**
   - Algorithm cari values m dan b yang minimize total squared error
   - Menggunakan calculus (derivative = 0) untuk find optimal point
   - Closed-form solution (tidak perlu iterative training)

3. **Assumptions:**
   - **Linearity**: Relationship antar X dan Y linear
   - **Independence**: Observations saling independent  
   - **Homoscedasticity**: Constant variance of errors
   - **Normality**: Errors terdistribusi normal

#### **Kapan Linear Regression Bekerja Baik:**
✅ Relationship truly linear  
✅ Limited noise in data
✅ Sufficient data points
✅ No extreme outliers

#### **Kapan Linear Regression Struggle:**
❌ Non-linear relationships
❌ High-dimensional data
❌ Complex feature interactions  
❌ Time-series dengan trends

### 1.5 Latihan dan Eksperimen

#### **Eksperimen 1: Effect of Noise**
```python
noise_levels = [0, 100, 300, 500]
r2_scores = []

for noise in noise_levels:
    # Generate data dengan different noise levels
    y_noisy = ukuran_rumah * 5 + np.random.randint(-noise, noise+1, n_samples)
    y_noisy = np.maximum(y_noisy, 100)
    
    # Reshape dan split
    X_reshaped = ukuran_rumah.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X_reshaped, y_noisy, test_size=0.2, random_state=42)
    
    # Train dan evaluate
    temp_model = LinearRegression()
    temp_model.fit(X_tr, y_tr)
    y_temp_pred = temp_model.predict(X_te)
    r2 = r2_score(y_te, y_temp_pred)
    r2_scores.append(r2)
    
    print(f"Noise level {noise}: R² = {r2:.3f}")

# Plot hasil
plt.figure(figsize=(8, 5))
plt.plot(noise_levels, r2_scores, marker='o', linewidth=2)
plt.xlabel('Noise Level')
plt.ylabel('R² Score') 
plt.title('Impact of Noise on Model Performance')
plt.grid(True, alpha=0.3)
plt.show()
```

#### **Eksperimen 2: Effect of Sample Size**
```python
sample_sizes = [20, 50, 100, 200, 500]
r2_scores_size = []

for size in sample_sizes:
    # Generate data dengan different sample sizes
    ukuran_temp = np.random.randint(50, 301, size)
    harga_temp = ukuran_temp * 5 + np.random.randint(-200, 201, size)
    harga_temp = np.maximum(harga_temp, 100)
    
    X_temp = ukuran_temp.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X_temp, harga_temp, test_size=0.2, random_state=42)
    
    temp_model = LinearRegression()
    temp_model.fit(X_tr, y_tr)
    y_temp_pred = temp_model.predict(X_te)
    r2 = r2_score(y_te, y_temp_pred)
    r2_scores_size.append(r2)
    
    print(f"Sample size {size}: R² = {r2:.3f}")

# Plot hasil  
plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, r2_scores_size, marker='s', linewidth=2, color='orange')
plt.xlabel('Sample Size')
plt.ylabel('R² Score')
plt.title('Impact of Sample Size on Model Performance')  
plt.grid(True, alpha=0.3)
plt.show()
```

### 1.6 Key Takeaways

🎯 **Konsep Utama:**
- Linear Regression memodelkan relationship linear antar variables
- Training = mencari best fit line yang minimize errors
- Evaluation menggunakan MSE, RMSE, dan R-squared metrics

🔧 **Practical Skills:**
- Data preparation dan reshaping untuk scikit-learn
- Train-test split untuk unbiased evaluation
- Visualization untuk understand data dan results

📊 **Insights:**
- More data → generally better performance (sampai titik tertentu)
- More noise → worse performance (obvious)
- Linear models simple tapi powerful untuk many real-world problems

🚀 **Next Steps:**
- Experiment dengan real datasets (housing prices, etc.)
- Try multiple features (size, location, age, etc.)  
- Learn about regularization (Ridge, Lasso) untuk handle overfitting

---

## MODUL 2: CLASSIFICATION 

### 2.1 Konsep Dasar Classification

**Classification** berbeda dengan regression karena kita prediksi **kategori/kelas** bukan angka kontinyu.

**Contoh Classification Problems:**
- Medical: Tumor → Benign atau Malignant
- Finance: Transaction → Fraud atau Legitimate  
- Technology: Image → Cat, Dog, atau Bird
- Business: Customer → Will Buy atau Won't Buy

**Jenis Classification:**
1. **Binary Classification**: 2 kelas (Ya/Tidak)
2. **Multiclass Classification**: >2 kelas (A/B/C/...)  
3. **Multilabel Classification**: Multiple labels per instance

### 2.2 Decision Tree Algorithm

**Mengapa Decision Tree?**
- **Interpretable**: Bisa dijelaskan dengan mudah (if-then rules)
- **No assumptions**: Tidak perlu assume distribusi data
- **Handle mixed data**: Numerical dan categorical features
- **Feature selection**: Otomatis pilih features yang penting

**Cara Kerja Decision Tree:**
```
Algoritma recursively split data untuk maximize information gain.

Pseudocode:
1. Start dengan semua data di root node
2. Find best feature + threshold untuk split data  
3. Split data menjadi subset berdasarkan condition
4. Repeat untuk setiap subset sampai stopping criteria
5. Stopping: pure nodes, max depth, min samples, etc.
```

### 2.3 Hands-On: Iris Classification

#### **Dataset Overview:**
Iris dataset adalah dataset klasik dalam ML, berisi:
- **150 samples** dari 3 spesies iris flowers
- **4 features**: sepal length/width, petal length/width  
- **3 classes**: Setosa, Versicolor, Virginica

#### **Step 1: Load dan Explore Data**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("✅ Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Features: {list(iris.feature_names)}")  
print(f"Classes: {list(iris.target_names)}")

# Basic statistics
print(f"\n📊 Dataset Overview:")
print(df.describe())
print(f"\n🎯 Class distribution:")
print(df['species'].value_counts())
```

#### **Step 2: Data Visualization**
```python
# Setup plotting
plt.figure(figsize=(15, 10))

# Plot 1: Distribution of each feature by species
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 3, i+1)
    for species_idx, species_name in enumerate(iris.target_names):
        data_subset = df[df['species'] == species_idx][feature]
        plt.hist(data_subset, alpha=0.7, label=species_name, bins=15)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution: {feature}')
    plt.legend()

# Plot 2: Correlation heatmap
plt.subplot(2, 3, 5)
correlation_matrix = df.iloc[:, :-1].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation')

# Plot 3: Scatter plot matrix (petal features)
plt.subplot(2, 3, 6)
for species_idx, species_name in enumerate(iris.target_names):
    subset = df[df['species'] == species_idx]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'], 
               label=species_name, alpha=0.7)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Width')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**💡 Expected Insights:**
- **Setosa clearly separable** dari yang lain (especially pada petal features)
- **Versicolor dan Virginica overlap** lebih banyak
- **Petal features more discriminative** dibanding sepal features
- **Strong correlation** between petal length dan width

#### **Step 3: Prepare Data**
```python
# Separate features dan target
X = iris.data
y = iris.target

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train-test split dengan stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Check class distribution dalam splits
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

print(f"\n📊 Training distribution:")
for i, count in enumerate(counts_train):
    print(f"  {iris.target_names[i]}: {count}")
    
print(f"📊 Testing distribution:")
for i, count in enumerate(counts_test):
    print(f"  {iris.target_names[i]}: {count}")
```

**💡 Mengapa Stratify?**
- Ensure proportional representation dari setiap class
- Prevent bias dalam train/test splits
- Especially important untuk imbalanced datasets

#### **Step 4: Train Decision Tree**
```python
# Create dan train model
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=3,  # Limit depth untuk prevent overfitting
    min_samples_split=2,
    min_samples_leaf=1
)

# Training
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Feature importance
feature_importance = model.feature_importances_
feature_names = iris.feature_names

print(f"\n📊 Feature Importance:")
for name, importance in zip(feature_names, feature_importance):
    print(f"  {name}: {importance:.3f}")

# Identify most important feature
most_important_idx = np.argmax(feature_importance)
print(f"✨ Most important: {feature_names[most_important_idx]}")
```

**💡 Feature Importance Explained:**
- **Based on information gain** dari each feature
- **Higher values** = more useful untuk classification  
- **Sum to 1.0** across all features
- **Decision tree automatically** performs feature selection

#### **Step 5: Make Predictions dan Evaluate**
```python
# Predictions
y_pred = model.predict(X_test)

# Basic accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

# Detailed classification report
print(f"\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n🎯 Confusion Matrix:")
print(cm)
```

**💡 Understanding Classification Report:**

```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        15
  versicolor       1.00      0.93      0.97        15  
   virginica       0.93      1.00      0.97        15

Interpretation:
- Setosa: Perfect classification (as expected)
- Versicolor: 100% precision, 93% recall (missed 1 instance)
- Virginica: 93% precision, 100% recall (1 false positive)
```

#### **Step 6: Visualize Decision Tree**
```python
plt.figure(figsize=(20, 10))

# Plot decision tree structure
tree.plot_tree(model, 
               feature_names=iris.feature_names,
               class_names=iris.target_names, 
               filled=True,
               rounded=True,
               fontsize=10)

plt.title('Decision Tree for Iris Classification')
plt.show()

# Text representation  
print(f"\n🌳 Decision Tree Rules:")
tree_rules = tree.export_text(model, feature_names=iris.feature_names)
print(tree_rules)
```

**💡 Reading the Tree:**
- **Root node**: First split (petal width ≤ 0.8)
- **Internal nodes**: Decision points dengan conditions  
- **Leaf nodes**: Final predictions dengan class
- **Gini impurity**: Measure of "purity" (0 = pure, 0.5 = mixed)
- **Samples**: Number of training samples reaching node

#### **Step 7: Cross Validation**
```python
# 5-Fold Cross Validation untuk robust evaluation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"\n🔄 Cross Validation Results:")
for i, score in enumerate(cv_scores):
    print(f"  Fold {i+1}: {score:.3f}")

print(f"\n📊 CV Summary:")
print(f"  Mean accuracy: {cv_scores.mean():.3f}")
print(f"  Standard deviation: {cv_scores.std():.3f}")  
print(f"  95% confidence interval: {cv_scores.mean():.3f} ± {1.96*cv_scores.std():.3f}")

# Performance interpretation
if cv_scores.mean() > 0.95:
    print("🏆 Excellent performance!")
elif cv_scores.mean() > 0.90:
    print("🎉 Very good performance!")  
elif cv_scores.mean() > 0.80:
    print("👍 Good performance!")
else:
    print("📈 Room for improvement")
```

#### **Step 8: Interactive Prediction**
```python
def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict iris species berdasarkan measurements
    """
    # Make prediction
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get species name
    species_name = iris.target_names[prediction]
    confidence = probabilities[prediction]
    
    print(f"\n🌸 Iris Species Prediction:")
    print(f"Input: SL={sepal_length}, SW={sepal_width}, PL={petal_length}, PW={petal_width}")
    print(f"Predicted species: {species_name}")
    print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
    
    print(f"\nAll probabilities:")
    for i, (species, prob) in enumerate(zip(iris.target_names, probabilities)):
        indicator = "🎯" if i == prediction else "  "
        print(f"{indicator} {species}: {prob:.3f} ({prob*100:.1f}%)")
    
    return species_name, confidence

# Test dengan beberapa contoh
test_cases = [
    (5.1, 3.5, 1.4, 0.2),  # Typical Setosa
    (6.2, 2.9, 4.3, 1.3),  # Typical Versicolor  
    (7.3, 2.9, 6.3, 1.8),  # Typical Virginica
]

for i, measurements in enumerate(test_cases):
    print(f"\n{'='*50}")
    print(f"Test Case {i+1}:")
    predict_iris_species(*measurements)
```

### 2.4 Mengapa Decision Tree Bekerja?

#### **Information Theory Background:**

1. **Entropy**: Measure of "impurity" dalam dataset
   ```
   Entropy(S) = -Σ(p_i * log2(p_i))
   
   Where:
   - p_i = proportion of samples belonging to class i
   - Pure set (one class): Entropy = 0
   - Maximum impurity (equal classes): Entropy = log2(n_classes)
   ```

2. **Information Gain**: Reduction in entropy after split
   ```
   IG(S, F) = Entropy(S) - Σ((|S_v|/|S|) * Entropy(S_v))
   
   Where:
   - S = original set
   - F = feature used untuk split  
   - S_v = subset setelah split by feature F
   ```

3. **Gini Impurity**: Alternative to entropy (faster to compute)
   ```
   Gini(S) = 1 - Σ(p_i)²
   
   Interpretation sama dengan entropy, tapi different formula
   ```

#### **Decision Tree Construction Process:**

```
1. Start dengan root node containing all training data
2. For each possible split (feature + threshold):
   a. Calculate information gain
   b. Keep track of best split
3. Apply best split, create child nodes
4. Recursively repeat untuk each child node
5. Stop when any stopping criteria met:
   - Node is pure (single class)
   - Maximum depth reached
   - Minimum samples per node  
   - No significant information gain
```

#### **Advantages dan Disadvantages:**

**✅ Advantages:**
- **Interpretable**: Easy to understand dan explain
- **No data preprocessing**: Handle missing values, mixed datatypes
- **Feature selection**: Implicitly selects important features
- **Non-linear relationships**: Can capture complex patterns
- **Fast prediction**: O(log n) complexity

**❌ Disadvantages:**  
- **Overfitting prone**: Especially dengan deep trees
- **Instability**: Small data changes → completely different tree
- **Bias**: Favor features dengan more levels
- **Limited expressiveness**: Axis-aligned splits only

#### **Best Practices:**

1. **Hyperparameter Tuning:**
   - `max_depth`: Typical range 3-10
   - `min_samples_split`: 2-50  
   - `min_samples_leaf`: 1-20
   - `min_impurity_decrease`: 0.0-0.1

2. **Prevent Overfitting:**
   - Use validation set untuk choose hyperparameters
   - Cross-validation untuk robust evaluation  
   - Ensemble methods (Random Forest, XGBoost)
   - Pruning techniques

3. **Data Considerations:**
   - **Feature scaling**: Not required (unlike KNN, SVM)
   - **Missing values**: Can be handled directly
   - **Categorical variables**: Need proper encoding
   - **Imbalanced classes**: Consider class weights

### 2.5 Latihan dan Advanced Topics

#### **Eksperimen 1: Effect of Tree Depth**
```python
depths = range(1, 11)
train_scores = []
test_scores = []

for depth in depths:
    # Train model dengan different depths
    temp_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    temp_model.fit(X_train, y_train)
    
    # Score pada training dan testing set
    train_score = temp_model.score(X_train, y_train)
    test_score = temp_model.score(X_test, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"Depth {depth}: Train={train_score:.3f}, Test={test_score:.3f}")

# Plot overfitting curve
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, label='Training Accuracy', marker='o')
plt.plot(depths, test_scores, label='Testing Accuracy', marker='s')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Overfitting Analysis: Training vs Testing Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**💡 Expected Pattern:**
- **Training accuracy** increases dengan depth
- **Testing accuracy** increases dulu, then decreases (overfitting)
- **Sweet spot** around depth 3-5 untuk iris dataset

#### **Eksperimen 2: Feature Importance Analysis**
```python
# Train multiple trees dengan different random states
n_runs = 100
feature_importances = []

for i in range(n_runs):
    temp_model = DecisionTreeClassifier(max_depth=3, random_state=i)
    temp_model.fit(X_train, y_train)
    feature_importances.append(temp_model.feature_importances_)

# Convert ke array untuk easier manipulation
feature_importances = np.array(feature_importances)

# Calculate statistics
mean_importance = np.mean(feature_importances, axis=0)
std_importance = np.std(feature_importances, axis=0)

# Visualization
plt.figure(figsize=(10, 6))
position = np.arange(len(iris.feature_names))
plt.bar(position, mean_importance, yerr=std_importance, 
        capsize=5, alpha=0.7, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance (Mean ± Std across 100 runs)')
plt.xticks(position, iris.feature_names, rotation=45)
plt.grid(True, alpha=0.3)

# Print rankings
print("📊 Average Feature Ranking:")
for i in np.argsort(-mean_importance):
    print(f"{i+1}. {iris.feature_names[i]}: {mean_importance[i]:.3f} ± {std_importance[i]:.3f}")

plt.show()
```

### 2.6 Key Takeaways

🎯 **Konsep Utama:**
- Classification prediksi categories, bukan continuous values
- Decision Tree membangun rules berdasarkan information gain
- Feature importance otomatis calculated dan bisa diinterpret

🔧 **Practical Skills:**  
- Stratified train-test split untuk balanced evaluation
- Cross-validation untuk robust performance assessment
- Confusion matrix dan classification report interpretation

📊 **Insights:**
- Simple algoritma bisa achieve excellent performance
- Interpretability vs accuracy trade-off adalah penting
- Visualization crucial untuk understand decision boundaries

🚀 **Next Steps:**
- Try ensemble methods (Random Forest, XGBoost)  
- Experiment dengan imbalanced datasets
- Learn other classification algorithms (SVM, KNN, Naive Bayes)

---

## MODUL 3: NEURAL NETWORK (DEEP LEARNING)

### 3.1 Apa itu Neural Network?

Neural Network (NN) adalah model yang terinspirasi dari cara kerja neuron di otak.
Kalau linear regression itu seperti “garis lurus”, neural network itu seperti “tumpukan lapisan” yang bisa mempelajari pola jauh lebih kompleks.

**Intuisi sederhana:**
- Data masuk → diproses beberapa kali (layer) → keluar prediksi
- Setiap layer belajar “fitur” yang makin abstrak

### 3.2 Komponen Penting

1. **Neuron / Unit**
    - Mengambil input, mengalikan dengan bobot, menambah bias
2. **Weights & Biases**
    - Parameter yang dipelajari
3. **Activation Function**
    - Membuat model bisa belajar pola non-linear
    - Contoh: ReLU, Sigmoid, Softmax
4. **Loss Function**
    - Mengukur seberapa salah prediksi
5. **Optimizer**
    - Cara memperbarui weights (contoh: SGD, Adam)
6. **Epoch & Batch**
    - Epoch: 1 kali “keliling” seluruh data
    - Batch: potongan data kecil saat training

### 3.3 Mengapa Deep Learning Perlu Banyak Data?

Karena parameter NN sangat banyak. Semakin banyak parameter, semakin besar kemampuan model, tetapi semakin besar juga kebutuhan data agar tidak overfitting.

### 3.4 Praktik: MNIST Digit Recognition

Program latihan ada di file `03_neural_network.py`.

**Yang akan Anda pelajari dari program ini:**
- Membaca dataset gambar angka (0–9)
- Membuat model `Sequential`
- Training + melihat metrik akurasi
- Mencoba prediksi pada gambar yang belum pernah dilihat

**Cara menjalankan (paling aman):**

Windows PowerShell (gunakan Python dari venv project):

```bash
& "./.venv/Scripts/python.exe" 03_neural_network.py
```

Mac/Linux (kalau venv sudah aktif):

```bash
python 03_neural_network.py
```

### 3.5 Mengapa Jawabannya Seperti Itu? (Penjelasan Inti)

Pada MNIST, output model biasanya berupa **probabilitas 10 kelas** (0–9) menggunakan Softmax.

- Model menghasilkan vector seperti: `[0.01, 0.00, 0.02, ..., 0.93]`
- Angka tertinggi adalah kelas prediksi
- “Confidence” bisa dibaca dari nilai probabilitas tertinggi

**Kenapa bisa benar?**
Karena selama training, model menyesuaikan weights untuk meminimalkan loss.
Proses ini dilakukan berulang (epoch) menggunakan backpropagation.

### 3.6 Latihan Wajib (Agar Paham)

1. Ubah `epochs` (misalnya 1, 3, 10) lalu bandingkan akurasi.
2. Ubah ukuran batch (misalnya 16 vs 128) dan perhatikan training time.
3. Tambahkan `Dropout` lalu lihat efeknya.
4. Catat: kapan model mulai overfitting? (train accuracy naik, val accuracy stagnan/turun).

---

## MODUL 4: COMPUTER VISION (AI YANG MELIHAT)

### 4.1 Apa itu Computer Vision?

Computer Vision (CV) adalah cabang AI untuk memahami informasi visual (gambar/video).
Berbeda dari MNIST yang sederhana, CV di dunia nyata menghadapi:
- Variasi cahaya, posisi, skala
- Background beragam
- Noise dan blur

### 4.2 Kenapa CNN (Convolutional Neural Network) Sangat Penting?

CNN unggul untuk gambar karena:
- **Convolution** menangkap pola lokal (tepi, tekstur)
- **Pooling** mereduksi ukuran sambil mempertahankan informasi penting
- Layer awal belajar tepi/garis, layer tengah belajar bentuk, layer akhir belajar objek

### 4.3 Praktik: CIFAR-10

Program latihan ada di file `04_computer_vision.py`.

**Yang akan Anda pelajari:**
- Dataset gambar warna 32×32 (10 kelas)
- CNN sederhana untuk klasifikasi
- Evaluasi akurasi dan visualisasi prediksi

**Cara menjalankan (paling aman):**

```bash
& "./.venv/Scripts/python.exe" 04_computer_vision.py
```

### 4.4 Mengapa Model CNN Kadang Lambat?

Karena operasi convolution cukup berat secara komputasi.
Jika Anda tidak punya GPU, training tetap bisa berjalan di CPU tetapi akan lebih lambat.

**Tips aman untuk pemula (tanpa ubah konsep):**
- Kurangi epoch
- Kurangi jumlah filter di layer CNN
- Gunakan subset dataset dulu untuk eksperimen

### 4.5 Latihan Wajib

1. Ubah jumlah filter (misalnya 16 → 32) lalu bandingkan akurasi.
2. Tambahkan 1 layer convolution lagi; lihat apakah membaik atau overfitting.
3. Coba augmentasi sederhana (flip/rotate) jika sudah paham.

---

## BAGIAN 8: MENGAPA ALGORITMA “BELAJAR”?

### 8.1 Konsep Besar: Optimisasi

Hampir semua ML (termasuk deep learning) intinya:

1. Tentukan **fungsi salah** (loss)
2. Cari parameter model yang membuat loss sekecil mungkin

Untuk model linear, kadang ada solusi “langsung” (closed-form).
Untuk neural network, biasanya perlu proses iteratif (gradient descent).

### 8.2 Gradient Descent Secara Intuisi

Bayangkan Anda berada di bukit berkabut dan ingin turun ke titik terendah.
- Anda cek arah paling menurun → melangkah sedikit
- Ulangi sampai ketemu lembah

**Learning rate** adalah ukuran langkah.
- Terlalu besar: bisa “loncat-loncat” dan tidak stabil
- Terlalu kecil: terlalu lama sampai

---

## BAGIAN 9: TROUBLESHOOTING & PROBLEM SOLVING

### 9.1 “KeyboardInterrupt” Saat Grafik Muncul — Error atau Normal?

Ini sering membingungkan pemula.

**Kasus umum:**
- Program menampilkan grafik (`plt.show()`)
- Anda menutup jendela grafik / menekan Ctrl+C
- Terminal menampilkan `KeyboardInterrupt`

**Maknanya:**
- Biasanya *bukan bug model*, tapi Anda menghentikan eksekusi secara manual.

**Cara menjalankan yang benar:**
- Saat grafik muncul: lihat grafik → tutup jendela → program lanjut ke step berikutnya.

### 9.2 Kalau Program “Diam” Lama

Kemungkinan:
- Sedang download dataset (MNIST/CIFAR)
- Sedang training model

Solusi:
- Tunggu 1–5 menit pertama (download)
- Kurangi epoch
- Pastikan Anda menjalankan via `.venv` agar dependency benar

### 9.3 Error Import TensorFlow / OpenCV

Jika muncul error import:
- Pastikan menjalankan Python dari `.venv`
- Jangan campur Python global dengan venv

Windows (cek interpreter):

```bash
& "./.venv/Scripts/python.exe" -c "import sys; print(sys.executable)"
```

---

## BAGIAN 10: NEXT LEVEL (PROJECT IDEAS & CAREER PATH)

### 10.1 Project Ideas (Pemula → Menengah)

1. **Prediksi harga motor bekas** (regression)
    - Fitur: tahun, jarak tempuh, merek, kondisi
2. **Klasifikasi kualitas buah** (classification)
    - Data tabular (berat, diameter, warna) atau image sederhana
3. **Digit recognizer dari tulisan sendiri**
    - Ambil gambar dari kamera / file
4. **Deteksi masker** (computer vision)
    - Dataset kecil + transfer learning (tahap lanjut)

### 10.2 Alur Belajar Menjadi AI Developer

1. Python dasar (sudah)
2. Numpy, Pandas, Matplotlib
3. ML dasar (regression/classification)
4. Evaluation, data leakage, cross-validation
5. Deep learning dasar (NN, CNN)
6. Project portfolio + deployment sederhana

---

## LAMPIRAN: ALUR LATIHAN YANG DIREKOMENDASIKAN

### Hari 1–2
- Jalankan `01_linear_regression.py`
- Catat: apa itu fitur, target, train-test split, R²

### Hari 3–4
- Jalankan `02_classification.py`
- Catat: confusion matrix, precision/recall, overfitting (depth)

### Hari 5–7
- Jalankan `03_neural_network.py`
- Catat: epoch, batch, loss vs accuracy

### Minggu 2
- Jalankan `04_computer_vision.py`
- Catat: CNN vs dense, training time, generalisasi

---

## PENUTUP

Kalau Anda konsisten menyelesaikan 4 modul ini sambil eksperimen, Anda sudah punya fondasi kuat untuk naik ke:
- Random Forest / XGBoost
- Feature engineering
- Transfer learning (ResNet/MobileNet)
- Model deployment (FastAPI/Streamlit)

Selanjutnya, yang paling penting adalah: **punya project nyata**.