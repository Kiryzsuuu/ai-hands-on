"""
🧠 AI PROGRAM 3: NEURAL NETWORK - JARINGAN SARAF TIRUAN

Program ini mengajarkan konsep dasar Neural Network (Deep Learning).
Kita akan membuat jaringan saraf untuk mengenali angka tulisan tangan (MNIST).

Konsep yang dipelajari:
- Apa itu Neural Network?
- Layers, Neurons, Weights & Biases
- Forward Propagation & Backpropagation
- Training dengan Gradient Descent
- Deep Learning
"""

# =================== IMPORT LIBRARIES ===================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("🧠 Selamat Datang di Pembelajaran AI - Neural Network!")
print("=" * 60)

# =================== STEP 1: PENGENALAN NEURAL NETWORK ===================
print("\n🎯 STEP 1: Apa itu Neural Network?")
print("""
Neural Network adalah model AI yang terinspirasi dari otak manusia:

🧠 STRUKTUR NEURAL NETWORK:
   Input Layer → Hidden Layer(s) → Output Layer
   
   Neuron = Unit pemrosesan yang menerima input, 
            melakukan perhitungan, dan menghasilkan output
            
   Weight = Bobot yang menentukan seberapa penting suatu input
   Bias = Nilai tambahan yang membantu model belajar
   
🔄 CARA KERJA:
   1. Forward Propagation: Data mengalir dari input ke output
   2. Loss Calculation: Hitung error prediksi
   3. Backpropagation: Update weight untuk mengurangi error
   4. Repeat: Ulangi sampai model akurat
""")

input("Tekan Enter untuk melanjutkan...")

# =================== STEP 2: LOAD DATASET MNIST ===================
print("\n📊 STEP 2: Memuat Dataset MNIST")
print("MNIST = Database angka tulisan tangan (0-9)")

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"✅ Dataset berhasil dimuat!")
print(f"   Training images: {X_train.shape}")
print(f"   Training labels: {y_train.shape}")
print(f"   Test images: {X_test.shape}")
print(f"   Test labels: {y_test.shape}")

# =================== STEP 3: EKSPLORASI DATA ===================
print("\n🔍 STEP 3: Eksplorasi Data")

# Tampilkan beberapa contoh gambar
plt.figure(figsize=(12, 8))

print("Menampilkan contoh angka tulisan tangan:")
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')

plt.suptitle('Contoh Dataset MNIST - Angka Tulisan Tangan')
plt.tight_layout()
plt.show()

# Analisis distribusi data
unique, counts = np.unique(y_train, return_counts=True)
print(f"\n📈 Distribusi angka dalam training set:")
for digit, count in zip(unique, counts):
    print(f"   Angka {digit}: {count} gambar")

print(f"\n📏 Info gambar:")
print(f"   Ukuran: {X_train[0].shape} pixels")
print(f"   Rentang nilai: {X_train.min()} - {X_train.max()}")
print(f"   Tipe data: {X_train.dtype}")

# =================== STEP 4: PREPROCESSING DATA ===================
print("\n🔧 STEP 4: Preprocessing Data")

# Normalisasi: ubah pixel values dari 0-255 menjadi 0-1
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

# Reshape: dari (28, 28) menjadi (784,) - flatten
X_train_flat = X_train_norm.reshape(X_train_norm.shape[0], 28*28)
X_test_flat = X_test_norm.reshape(X_test_norm.shape[0], 28*28)

# Convert labels ke categorical (one-hot encoding)
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print(f"✅ Preprocessing selesai:")
print(f"   X_train shape: {X_train_flat.shape}")
print(f"   y_train shape: {y_train_cat.shape}")
print(f"   Pixel range: {X_train_norm.min():.1f} - {X_train_norm.max():.1f}")

# Tampilkan contoh one-hot encoding
print(f"\n🔢 Contoh One-Hot Encoding:")
print(f"   Label asli: {y_train[0]}")
print(f"   One-hot: {y_train_cat[0]}")

# =================== STEP 5: MEMBANGUN NEURAL NETWORK ===================
print("\n🧠 STEP 5: Membangun Neural Network Architecture")

print("""
🏗️ ARSITEKTUR YANG AKAN DIBUAT:
   Input Layer: 784 neurons (28×28 pixels)
   Hidden Layer 1: 128 neurons + ReLU activation
   Hidden Layer 2: 64 neurons + ReLU activation  
   Output Layer: 10 neurons (0-9 digits) + Softmax
""")

# Buat model Sequential
model = keras.Sequential([
    # Hidden Layer 1
    layers.Dense(128, activation='relu', input_shape=(784,), name='hidden_1'),
    
    # Hidden Layer 2
    layers.Dense(64, activation='relu', name='hidden_2'),
    
    # Output Layer
    layers.Dense(10, activation='softmax', name='output')
])

# Compile model
model.compile(
    optimizer='adam',      # Algoritma optimisasi
    loss='categorical_crossentropy',  # Fungsi loss untuk klasifikasi
    metrics=['accuracy']   # Metrik evaluasi
)

# Tampilkan summary model
print("\n📋 Model Architecture:")
model.summary()

# Hitung total parameters
total_params = model.count_params()
print(f"\n🔢 Total Parameters: {total_params:,}")
print(f"   Parameter = Weight + Bias yang akan dipelajari oleh model")

# =================== STEP 6: TRAINING NEURAL NETWORK ===================
print("\n🔄 STEP 6: Training Neural Network")
print("Proses pembelajaran model dari data...")

# Callback untuk monitoring training
class TrainingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 2 == 0:
            print(f"   Epoch {epoch+1}: loss={logs['loss']:.4f}, accuracy={logs['accuracy']:.4f}")

# Training model
print("🔥 Memulai training...")
history = model.fit(
    X_train_flat, y_train_cat,
    epochs=10,              # Jumlah iterasi training
    batch_size=128,         # Jumlah sample per batch
    validation_split=0.1,   # 10% data untuk validasi
    verbose=0,              # Tidak tampilkan detail per batch
    callbacks=[TrainingCallback()]
)

print("✅ Training selesai!")

# =================== STEP 7: EVALUASI MODEL ===================
print("\n📊 STEP 7: Evaluasi Performa Model")

# Evaluasi pada test set
test_loss, test_accuracy = model.evaluate(X_test_flat, y_test_cat, verbose=0)

print(f"🎯 Hasil Evaluasi pada Test Set:")
print(f"   Loss: {test_loss:.4f}")
print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Prediksi pada test set
predictions = model.predict(X_test_flat, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# Hitung akurasi per digit
print(f"\n📈 Akurasi per digit:")
for digit in range(10):
    mask = y_test == digit
    if np.sum(mask) > 0:
        digit_accuracy = np.mean(predicted_classes[mask] == y_test[mask])
        print(f"   Digit {digit}: {digit_accuracy:.3f} ({digit_accuracy*100:.1f}%)")

# =================== STEP 8: VISUALISASI TRAINING HISTORY ===================
print("\n📈 STEP 8: Visualisasi Training History")

plt.figure(figsize=(15, 5))

# Plot 1: Training & Validation Loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Training & Validation Accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Contoh prediksi benar dan salah
plt.subplot(1, 3, 3)
correct_indices = np.where(predicted_classes == y_test)[0]
incorrect_indices = np.where(predicted_classes != y_test)[0]

# Tampilkan beberapa prediksi salah (jika ada)
if len(incorrect_indices) > 0:
    idx = incorrect_indices[0]
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(f'Salah: Prediksi {predicted_classes[idx]}, Asli {y_test[idx]}')
else:
    idx = correct_indices[0]
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(f'Benar: Prediksi {predicted_classes[idx]}')
plt.axis('off')

plt.tight_layout()
plt.show()

# =================== STEP 9: ANALISIS PREDIKSI ===================
print("\n🔍 STEP 9: Analisis Prediksi Detail")

# Pilih beberapa contoh untuk analisis
sample_indices = [0, 1, 2, 3, 4]

plt.figure(figsize=(15, 6))

for i, idx in enumerate(sample_indices):
    # Gambar
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(f'Asli: {y_test[idx]}')
    plt.axis('off')
    
    # Probability distribution
    plt.subplot(2, 5, i+6)
    probs = predictions[idx]
    plt.bar(range(10), probs)
    plt.title(f'Prediksi: {predicted_classes[idx]}')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.xticks(range(10))

plt.suptitle('Analisis Prediksi: Gambar vs Probability Distribution')
plt.tight_layout()
plt.show()

# =================== STEP 10: VISUALISASI WEIGHTS ===================
print("\n🎨 STEP 10: Visualisasi Weights Layer Pertama")

# Ambil weights dari layer pertama
first_layer_weights = model.layers[0].get_weights()[0]  # Shape: (784, 128)

# Visualisasi beberapa neuron sebagai gambar
plt.figure(figsize=(12, 8))

print("Menampilkan weights 16 neuron pertama sebagai 'pola' yang dipelajari:")
for i in range(16):
    plt.subplot(4, 4, i+1)
    # Reshape weight neuron ke-i menjadi gambar 28x28
    weight_image = first_layer_weights[:, i].reshape(28, 28)
    plt.imshow(weight_image, cmap='RdBu')
    plt.title(f'Neuron {i+1}')
    plt.axis('off')

plt.suptitle('Weights Neuron - Pola yang Dipelajari Model')
plt.tight_layout()
plt.show()

print("💡 Setiap neuron belajar mengenali pola tertentu dalam gambar")

# =================== STEP 11: MODE INTERAKTIF ===================
print("\n🎮 STEP 11: Mode Interaktif - Test Prediksi!")

def predict_digit(model, image_index):
    """Fungsi untuk prediksi digit"""
    image = X_test[image_index]
    image_flat = X_test_flat[image_index:image_index+1]
    
    # Prediksi
    prediction = model.predict(image_flat, verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    return image, predicted_digit, confidence, prediction[0]

try:
    print("\n🎯 Test Neural Network dengan gambar random!")
    
    while True:
        print("\n" + "="*50)
        
        # Pilih gambar random atau input user
        choice = input("Pilih: (r)andom, (i)ndex tertentu, atau (q)uit: ").lower()
        
        if choice == 'q':
            break
        elif choice == 'r':
            idx = np.random.randint(0, len(X_test))
        elif choice == 'i':
            try:
                idx = int(input(f"Masukkan index (0-{len(X_test)-1}): "))
                if not 0 <= idx < len(X_test):
                    print("❌ Index tidak valid")
                    continue
            except ValueError:
                print("❌ Masukkan angka yang valid")
                continue
        else:
            print("❌ Pilihan tidak valid")
            continue
        
        # Prediksi
        image, predicted_digit, confidence, all_probs = predict_digit(model, idx)
        actual_digit = y_test[idx]
        
        # Tampilkan hasil
        plt.figure(figsize=(10, 4))
        
        # Gambar
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Index: {idx}')
        plt.axis('off')
        
        # Probability bar chart
        plt.subplot(1, 2, 2)
        colors = ['green' if i == predicted_digit else 'blue' for i in range(10)]
        plt.bar(range(10), all_probs, color=colors)
        plt.title('Probability Distribution')
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.xticks(range(10))
        
        plt.tight_layout()
        plt.show()
        
        # Print hasil
        print(f"\n🎯 HASIL PREDIKSI:")
        print(f"   Gambar aktual: {actual_digit}")
        print(f"   Prediksi model: {predicted_digit}")
        print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        
        if predicted_digit == actual_digit:
            print("   ✅ BENAR!")
        else:
            print("   ❌ SALAH!")
            
        # Top 3 predictions
        top3_indices = np.argsort(all_probs)[-3:][::-1]
        print(f"\n📊 Top 3 Prediksi:")
        for i, idx_pred in enumerate(top3_indices):
            print(f"   {i+1}. Digit {idx_pred}: {all_probs[idx_pred]:.3f} ({all_probs[idx_pred]*100:.1f}%)")
            
except KeyboardInterrupt:
    print("\n👋 Program dihentikan.")

print("\n" + "="*60)
print("🎓 SELAMAT! Anda telah menyelesaikan program Neural Network!")
print("🔥 Konsep yang telah Anda pelajari:")
print("   ✓ Arsitektur Neural Network (Input-Hidden-Output)")
print("   ✓ Forward Propagation dan Backpropagation")
print("   ✓ Training dengan Gradient Descent")
print("   ✓ Loss Function dan Optimization")
print("   ✓ Evaluasi model Deep Learning")
print("   ✓ Visualisasi weights dan training history")
print("   ✓ One-hot encoding dan normalisasi")
print("\n📚 Lanjutkan ke program berikutnya: 04_computer_vision.py")
print("="*60)