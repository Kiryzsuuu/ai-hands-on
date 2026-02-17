"""
🏠 AI PROGRAM 1: LINEAR REGRESSION - PREDIKSI HARGA RUMAH

Program ini mengajarkan konsep dasar Machine Learning dengan Linear Regression.
Kita akan memprediksi harga rumah berdasarkan ukuran rumah (luas dalam meter persegi).

Konsep yang dipelajari:
- Apa itu Machine Learning?
- Supervised Learning
- Linear Regression
- Data training dan testing
- Evaluasi model
"""

# =================== IMPORT LIBRARIES ===================
# Library untuk perhitungan matematika
import numpy as np
# Library untuk manipulasi data
import pandas as pd
# Library untuk membuat grafik
import matplotlib.pyplot as plt
# Library untuk machine learning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("🤖 Selamat Datang di Pembelajaran AI - Linear Regression!")
print("=" * 60)

# =================== STEP 1: BUAT DATA CONTOH ===================
print("\n📊 STEP 1: Membuat Data Contoh")
print("Kita akan buat data contoh rumah dengan ukuran dan harga")

# Data contoh: ukuran rumah (m²) dan harga (dalam jutaan rupiah)
np.random.seed(42)  # Untuk hasil yang konsisten setiap kali dijalankan

# Buat 100 data rumah dengan ukuran 50-300 m²
ukuran_rumah = np.random.randint(50, 301, 100)

# Harga = ukuran * 5 juta + noise (variasi alami)
# Rumus sederhana: harga ≈ ukuran × 5 juta + variasi ±500 juta
harga_rumah = ukuran_rumah * 5 + np.random.randint(-500, 501, 100)

# Pastikan harga tidak negatif
harga_rumah = np.maximum(harga_rumah, 100)

print(f"✅ Data dibuat: {len(ukuran_rumah)} rumah")
print(f"   Ukuran terkecil: {min(ukuran_rumah)}m²")
print(f"   Ukuran terbesar: {max(ukuran_rumah)}m²")
print(f"   Harga termurah: {min(harga_rumah)} juta")
print(f"   Harga termahal: {max(harga_rumah)} juta")

# =================== STEP 2: VISUALISASI DATA ===================
print("\n📈 STEP 2: Visualisasi Data")
print("Mari lihat hubungan antara ukuran dan harga rumah")

plt.figure(figsize=(10, 6))
plt.scatter(ukuran_rumah, harga_rumah, alpha=0.7, color='blue')
plt.xlabel('Ukuran Rumah (m²)')
plt.ylabel('Harga Rumah (Juta Rupiah)')
plt.title('Data Rumah: Ukuran vs Harga')
plt.grid(True, alpha=0.3)
plt.show()

print("✅ Dari grafik, terlihat ada hubungan positif:")
print("   Semakin besar rumah → Semakin mahal harga")

# =================== STEP 3: PERSIAPAN DATA ===================
print("\n🔧 STEP 3: Persiapan Data untuk Machine Learning")

# Reshape data untuk sklearn (harus 2D array)
X = ukuran_rumah.reshape(-1, 1)  # Input: ukuran rumah
y = harga_rumah                   # Output: harga rumah

print(f"Shape data X (input): {X.shape}")
print(f"Shape data y (output): {y.shape}")

# Split data menjadi training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Data training: {len(X_train)} rumah")
print(f"✅ Data testing: {len(X_test)} rumah")

# =================== STEP 4: MEMBUAT MODEL AI ===================
print("\n🤖 STEP 4: Membuat dan Melatih Model AI")
print("Kita menggunakan Linear Regression untuk belajar pola data")

# Buat model Linear Regression
model = LinearRegression()

# Latih model dengan data training
print("🔄 Melatih model...")
model.fit(X_train, y_train)

print("✅ Model berhasil dilatih!")
print(f"   Slope (kemiringan): {model.coef_[0]:.2f}")
print(f"   Intercept (titik potong): {model.intercept_:.2f}")
print(f"   Rumus: Harga = {model.coef_[0]:.2f} × Ukuran + {model.intercept_:.2f}")

# =================== STEP 5: EVALUASI MODEL ===================
print("\n📊 STEP 5: Evaluasi Performa Model")

# Prediksi pada data testing
y_pred = model.predict(X_test)

# Hitung akurasi
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Mean Squared Error: {mse:.2f}")
print(f"✅ R² Score: {r2:.2f}")
print(f"   R² menunjukkan seberapa baik model (0-1, semakin tinggi semakin baik)")

if r2 > 0.7:
    print("🎉 Model sangat baik!")
elif r2 > 0.5:
    print("👍 Model cukup baik!")
else:
    print("📚 Model perlu diperbaiki")

# =================== STEP 6: PREDIKSI BARU ===================
print("\n🔮 STEP 6: Coba Prediksi Harga Rumah Baru")

# Contoh prediksi untuk rumah dengan ukuran berbeda
ukuran_baru = [100, 150, 200, 250]

print("Prediksi harga untuk rumah dengan ukuran:")
for ukuran in ukuran_baru:
    prediksi_harga = model.predict([[ukuran]])[0]
    print(f"   🏠 {ukuran}m² → Rp {prediksi_harga:.0f} juta")

# =================== STEP 7: VISUALISASI HASIL ===================
print("\n📈 STEP 7: Visualisasi Hasil Prediksi")

# Buat garis prediksi
ukuran_range = np.linspace(50, 300, 100).reshape(-1, 1)
harga_prediksi_range = model.predict(ukuran_range)

plt.figure(figsize=(12, 8))

# Plot data asli
plt.subplot(2, 1, 1)
plt.scatter(X_test, y_test, alpha=0.7, color='blue', label='Data Asli')
plt.scatter(X_test, y_pred, alpha=0.7, color='red', label='Prediksi')
plt.plot(ukuran_range, harga_prediksi_range, color='green', linewidth=2, label='Model')
plt.xlabel('Ukuran Rumah (m²)')
plt.ylabel('Harga Rumah (Juta Rupiah)')
plt.title('Hasil Prediksi vs Data Asli')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot error
plt.subplot(2, 1, 2)
error = y_test - y_pred
plt.scatter(X_test, error, alpha=0.7, color='orange')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Ukuran Rumah (m²)')
plt.ylabel('Error (Selisih Prediksi)')
plt.title('Error Prediksi')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =================== STEP 8: INTERAKTIF ===================
print("\n🎮 STEP 8: Mode Interaktif - Coba Sendiri!")
print("Sekarang Anda bisa coba prediksi harga rumah dengan ukuran yang Anda inginkan")

try:
    while True:
        print("\n" + "="*50)
        ukuran_input = input("Masukkan ukuran rumah (m²) atau ketik 'exit' untuk keluar: ")
        
        if ukuran_input.lower() == 'exit':
            print("👋 Terima kasih sudah belajar AI!")
            break
            
        try:
            ukuran_input = float(ukuran_input)
            if ukuran_input <= 0:
                print("❌ Ukuran harus lebih dari 0")
                continue
                
            prediksi = model.predict([[ukuran_input]])[0]
            print(f"🏠 Prediksi harga rumah {ukuran_input}m²: Rp {prediksi:.0f} juta")
            
            # Berikan insight tambahan
            if ukuran_input < 100:
                print("💡 Tips: Rumah kecil, cocok untuk pasangan muda")
            elif ukuran_input < 200:
                print("💡 Tips: Ukuran sedang, cocok untuk keluarga kecil")
            else:
                print("💡 Tips: Rumah besar, cocok untuk keluarga besar")
                
        except ValueError:
            print("❌ Mohon masukkan angka yang valid")
            
except KeyboardInterrupt:
    print("\n👋 Program dihentikan. Terima kasih!")

print("\n" + "="*60)
print("🎓 SELAMAT! Anda telah menyelesaikan program Linear Regression!")
print("🔥 Konsep yang telah Anda pelajari:")
print("   ✓ Membuat dan mempersiapkan data")
print("   ✓ Melatih model Machine Learning")
print("   ✓ Evaluasi performa model")
print("   ✓ Membuat prediksi baru")
print("   ✓ Visualisasi hasil")
print("\n📚 Lanjutkan ke program berikutnya: 02_classification.py")
print("="*60)