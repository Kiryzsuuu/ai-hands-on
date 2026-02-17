"""
🌸 AI PROGRAM 2: CLASSIFICATION - KLASIFIKASI BUNGA IRIS

Program ini mengajarkan konsep Classification dalam Machine Learning.
Kita akan mengklasifikasikan jenis bunga iris berdasarkan ukuran kelopak dan mahkota.

Konsep yang dipelajari:
- Perbedaan Regression vs Classification
- Decision Tree Classifier
- Confusion Matrix
- Akurasi, Precision, Recall
- Cross Validation
"""

# =================== IMPORT LIBRARIES ===================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

print("🌸 Selamat Datang di Pembelajaran AI - Classification!")
print("=" * 60)

# =================== STEP 1: LOAD DATA IRIS ===================
print("\n📊 STEP 1: Memuat Dataset Iris")
print("Dataset Iris adalah dataset klasik untuk pembelajaran machine learning")

# Load dataset iris yang sudah ada di sklearn
iris = load_iris()

# Buat DataFrame untuk kemudahan
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("✅ Dataset berhasil dimuat!")
print(f"   Jumlah data: {len(df)} bunga")
print(f"   Jumlah fitur: {len(iris.feature_names)}")
print(f"   Jenis bunga: {list(iris.target_names)}")

# Tampilkan beberapa data pertama
print("\n📋 Contoh data:")
print(df.head())

# Info dataset
print(f"\n📈 Statistik dataset:")
print(df.describe())

# =================== STEP 2: EKSPLORASI DATA ===================
print("\n🔍 STEP 2: Eksplorasi Data Visual")

# Hitung distribusi species
species_count = df['species'].value_counts()
species_names = [iris.target_names[i] for i in species_count.index]

print("Distribusi jenis bunga:")
for i, name in enumerate(species_names):
    print(f"   {name}: {species_count[i]} bunga")

# Plot 1: Distribusi jenis bunga
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.pie(species_count.values, labels=species_names, autopct='%1.1f%%')
plt.title('Distribusi Jenis Bunga Iris')

# Plot 2: Petal Length vs Petal Width
plt.subplot(2, 3, 2)
for i, species in enumerate(iris.target_names):
    mask = df['species'] == i
    plt.scatter(df[mask]['petal length (cm)'], 
               df[mask]['petal width (cm)'], 
               label=species, alpha=0.7)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal: Length vs Width')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Sepal Length vs Sepal Width
plt.subplot(2, 3, 3)
for i, species in enumerate(iris.target_names):
    mask = df['species'] == i
    plt.scatter(df[mask]['sepal length (cm)'], 
               df[mask]['sepal width (cm)'], 
               label=species, alpha=0.7)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal: Length vs Width')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Histogram semua fitur
plt.subplot(2, 3, 4)
df.iloc[:, :-1].hist(bins=15, alpha=0.7, figsize=(8, 6))
plt.suptitle('Distribusi Semua Fitur')

# Plot 5: Boxplot
plt.subplot(2, 3, 5)
df.boxplot(column=['petal length (cm)', 'petal width (cm)'])
plt.title('Boxplot Petal Features')

# Plot 6: Correlation Heatmap
plt.subplot(2, 3, 6)
correlation = df.iloc[:, :-1].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Korelasi Antar Fitur')

plt.tight_layout()
plt.show()

print("✅ Dari visualisasi, kita bisa lihat:")
print("   • Setosa (biru) mudah dibedakan dari yang lain")
print("   • Versicolor (orange) dan Virginica (hijau) lebih sulit dibedakan")
print("   • Petal length dan width paling informatif untuk klasifikasi")

# =================== STEP 3: PERSIAPAN DATA ===================
print("\n🔧 STEP 3: Persiapan Data untuk Machine Learning")

# Pisahkan fitur (X) dan target (y)
X = iris.data  # Fitur: sepal/petal length/width
y = iris.target  # Target: jenis bunga (0, 1, 2)

print(f"Shape fitur (X): {X.shape}")
print(f"Shape target (y): {y.shape}")

# Split data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"✅ Data training: {len(X_train)} bunga")
print(f"✅ Data testing: {len(X_test)} bunga")

# Cek distribusi di training dan testing set
print("\nDistribusi training set:")
unique_train, counts_train = np.unique(y_train, return_counts=True)
for i, count in zip(unique_train, counts_train):
    print(f"   {iris.target_names[i]}: {count}")

print("\nDistribusi testing set:")
unique_test, counts_test = np.unique(y_test, return_counts=True)
for i, count in zip(unique_test, counts_test):
    print(f"   {iris.target_names[i]}: {count}")

# =================== STEP 4: MEMBUAT MODEL CLASSIFIER ===================
print("\n🤖 STEP 4: Membuat dan Melatih Model Classification")
print("Kita menggunakan Decision Tree untuk klasifikasi")

# Buat model Decision Tree
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,  # Batasi kedalaman untuk menghindari overfitting
    min_samples_split=2
)

# Latih model
print("🔄 Melatih model...")
model.fit(X_train, y_train)
print("✅ Model berhasil dilatih!")

# Tampilkan importance fitur
feature_importance = model.feature_importances_
print("\n📊 Pentingnya setiap fitur untuk klasifikasi:")
for i, importance in enumerate(feature_importance):
    print(f"   {iris.feature_names[i]}: {importance:.3f}")

# Fitur paling penting
most_important = np.argmax(feature_importance)
print(f"✨ Fitur paling penting: {iris.feature_names[most_important]}")

# =================== STEP 5: PREDIKSI DAN EVALUASI ===================
print("\n📊 STEP 5: Evaluasi Performa Model")

# Prediksi pada data testing
y_pred = model.predict(X_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Akurasi Model: {accuracy:.3f} ({accuracy*100:.1f}%)")

# Classification Report
print("\n📋 Laporan Klasifikasi Detail:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
print("\n🎯 Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualisasi Confusion Matrix
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')

# =================== STEP 6: CROSS VALIDATION ===================
print("\n🔄 STEP 6: Cross Validation")
print("Menguji model dengan berbagai pembagian data")

# 5-Fold Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5)

print("Skor Cross Validation:")
for i, score in enumerate(cv_scores):
    print(f"   Fold {i+1}: {score:.3f}")

print(f"\n✅ Rata-rata CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

if cv_scores.mean() > 0.9:
    print("🎉 Model sangat excellent!")
elif cv_scores.mean() > 0.8:
    print("👍 Model sangat baik!")
else:
    print("📚 Model perlu diperbaiki")

# =================== STEP 7: VISUALISASI DECISION TREE ===================
print("\n🌳 STEP 7: Visualisasi Decision Tree")

plt.subplot(1, 2, 2)
# Membuat plot decision tree (sederhana)
tree.plot_tree(model, 
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               filled=True,
               rounded=True,
               max_depth=3)  # Batasi untuk readability
plt.title('Decision Tree (Simplified)')

plt.tight_layout()
plt.show()

print("✅ Decision Tree menunjukkan aturan klasifikasi:")
print("   • Node akar: aturan pertama untuk membagi data")
print("   • Cabang: kondisi if-else")
print("   • Daun: prediksi akhir")

# =================== STEP 8: PREDIKSI INTERAKTIF ===================
print("\n🔮 STEP 8: Prediksi Bunga Baru")

# Beberapa contoh prediksi
print("Contoh prediksi bunga dengan ukuran berbeda:")

contoh_bunga = [
    [5.0, 3.0, 1.5, 0.3],  # Kemungkinan Setosa
    [6.0, 2.8, 4.0, 1.3],  # Kemungkinan Versicolor
    [7.0, 3.2, 5.5, 2.1]   # Kemungkinan Virginica
]

for i, bunga in enumerate(contoh_bunga):
    prediksi = model.predict([bunga])[0]
    probabilitas = model.predict_proba([bunga])[0]
    
    print(f"\n🌸 Bunga {i+1}:")
    print(f"   Ukuran: Sepal {bunga[0]}×{bunga[1]}cm, Petal {bunga[2]}×{bunga[3]}cm")
    print(f"   Prediksi: {iris.target_names[prediksi]}")
    print(f"   Confidence:")
    for j, prob in enumerate(probabilitas):
        print(f"     {iris.target_names[j]}: {prob:.3f} ({prob*100:.1f}%)")

# =================== STEP 9: MODE INTERAKTIF ===================
print("\n🎮 STEP 9: Mode Interaktif - Klasifikasi Bunga Anda!")

def get_input(prompt, min_val=0, max_val=10):
    """Helper function untuk input dengan validasi"""
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"❌ Nilai harus antara {min_val} dan {max_val}")
        except ValueError:
            print("❌ Mohon masukkan angka yang valid")

try:
    print("\nMasukkan ukuran bunga iris untuk klasifikasi:")
    print("(Rentang normal: Sepal 4-8cm, Petal 1-7cm)")
    
    while True:
        print("\n" + "="*50)
        print("🌸 KLASIFIKASI BUNGA IRIS BARU")
        
        # Input dari user
        sepal_length = get_input("Sepal Length (cm): ", 3, 9)
        sepal_width = get_input("Sepal Width (cm): ", 1, 5)
        petal_length = get_input("Petal Length (cm): ", 0.5, 7)
        petal_width = get_input("Petal Width (cm): ", 0.1, 3)
        
        # Prediksi
        input_bunga = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediksi = model.predict(input_bunga)[0]
        probabilitas = model.predict_proba(input_bunga)[0]
        
        # Tampilkan hasil
        print("\n" + "="*30)
        print("🎯 HASIL KLASIFIKASI:")
        print(f"Ukuran bunga: Sepal {sepal_length}×{sepal_width}cm, Petal {petal_length}×{petal_width}cm")
        print(f"\n🌸 PREDIKSI: {iris.target_names[prediksi].upper()}")
        
        print(f"\n📊 Tingkat Confidence:")
        max_prob_idx = np.argmax(probabilitas)
        for j, prob in enumerate(probabilitas):
            indicator = "🎯" if j == max_prob_idx else "  "
            print(f"{indicator} {iris.target_names[j]}: {prob:.3f} ({prob*100:.1f}%)")
        
        # Berikan insight
        print(f"\n💡 Insight:")
        if probabilitas[prediksi] > 0.8:
            print("   Model sangat yakin dengan prediksi ini!")
        elif probabilitas[prediksi] > 0.6:
            print("   Model cukup yakin, tapi ada kemungkinan lain")
        else:
            print("   Model tidak terlalu yakin, ukuran bunga ambigu")
            
        # Bandingkan dengan data training
        if prediksi == 0:  # Setosa
            print("   Setosa biasanya: Sepal kecil, Petal sangat kecil")
        elif prediksi == 1:  # Versicolor
            print("   Versicolor biasanya: Ukuran sedang")
        else:  # Virginica
            print("   Virginica biasanya: Sepal besar, Petal besar")
        
        # Tanya apakah mau lanjut
        lanjut = input("\nIngin coba lagi? (y/n): ").lower()
        if lanjut not in ['y', 'yes', 'ya']:
            break
            
except KeyboardInterrupt:
    print("\n👋 Program dihentikan. Terima kasih!")

print("\n" + "="*60)
print("🎓 SELAMAT! Anda telah menyelesaikan program Classification!")
print("🔥 Konsep yang telah Anda pelajari:")
print("   ✓ Load dan eksplorasi dataset")
print("   ✓ Classification dengan Decision Tree")
print("   ✓ Evaluasi model (Accuracy, Precision, Recall)")
print("   ✓ Confusion Matrix dan Cross Validation")
print("   ✓ Feature Importance")
print("   ✓ Prediksi dengan confidence level")
print("\n📚 Lanjutkan ke program berikutnya: 03_neural_network.py")
print("="*60)