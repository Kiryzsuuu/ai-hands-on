"""
👁️ AI PROGRAM 4: COMPUTER VISION - PENGLIHATAN KOMPUTER

Program ini mengajarkan konsep dasar Computer Vision dan CNN.
Kita akan membuat model untuk mengenali objek dalam gambar (CIFAR-10).

Konsep yang dipelajari:
- Computer Vision basics
- Convolutional Neural Network (CNN)
- Filters, Pooling, Feature Maps
- Image preprocessing
- Transfer Learning
"""

# =================== IMPORT LIBRARIES ===================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("👁️ Selamat Datang di Pembelajaran AI - Computer Vision!")
print("=" * 60)

# =================== STEP 1: PENGENALAN COMPUTER VISION ===================
print("\n🎯 STEP 1: Apa itu Computer Vision?")
print("""
Computer Vision = AI yang dapat "melihat" dan memahami gambar/video

🏗️ PERBEDAAN CNN vs REGULAR NEURAL NETWORK:

Regular NN:
   Input → Flatten → Dense Layers → Output
   
Convolutional NN (CNN):
   Input → Conv & Pooling → Flatten → Dense → Output

🔍 KOMPONEN CNN:
   • Convolutional Layer: Detect fitur (edges, shapes, patterns)
   • Pooling Layer: Reduce size, keep important info
   • Filters: Deteksi pola-pola spesifik
   • Feature Maps: Hasil deteksi fitur
   
💡 KEUNGGULAN CNN:
   • Lebih baik untuk gambar
   • Mendeteksi fitur lokal (edges, corners)
   • Parameter lebih sedikit
   • Translation invariant
""")

input("Tekan Enter untuk melanjutkan...")

# =================== STEP 2: LOAD DATASET CIFAR-10 ===================
print("\n📊 STEP 2: Memuat Dataset CIFAR-10")
print("CIFAR-10 = 60,000 gambar kecil dalam 10 kategori")

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

print(f"✅ Dataset berhasil dimuat!")
print(f"   Training images: {X_train.shape}")
print(f"   Training labels: {y_train.shape}")
print(f"   Test images: {X_test.shape}")
print(f"   Test labels: {y_test.shape}")

print(f"\n🏷️ Kategori objek:")
for i, name in enumerate(class_names):
    print(f"   {i}: {name}")

# =================== STEP 3: EKSPLORASI DATA ===================
print("\n🔍 STEP 3: Eksplorasi Data CIFAR-10")

# Tampilkan contoh gambar
plt.figure(figsize=(15, 10))

print("Menampilkan contoh gambar dari setiap kategori:")
for i in range(10):
    # Cari contoh dari setiap class
    class_indices = np.where(y_train == i)[0]
    sample_idx = class_indices[0]
    
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[sample_idx])
    plt.title(f'{class_names[i]}')
    plt.axis('off')

plt.suptitle('Contoh Dataset CIFAR-10 - Satu dari Setiap Kategori')
plt.tight_layout()
plt.show()

# Tampilkan variasi dalam satu class
plt.figure(figsize=(15, 6))
print(f"\nVariasi gambar dalam kategori '{class_names[0]}' (airplane):")
airplane_indices = np.where(y_train == 0)[0][:10]
for i, idx in enumerate(airplane_indices):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[idx])
    plt.title(f'Sample {i+1}')
    plt.axis('off')

plt.suptitle(f'Variasi dalam Kategori: {class_names[0]}')
plt.tight_layout()
plt.show()

# Analisis distribusi data
unique, counts = np.unique(y_train, return_counts=True)
print(f"\n📈 Distribusi data training:")
for class_idx, count in zip(unique, counts):
    print(f"   {class_names[class_idx]}: {count} gambar")

print(f"\n📏 Info gambar:")
print(f"   Ukuran: {X_train[0].shape} pixels")
print(f"   Rentang nilai: {X_train.min()} - {X_train.max()}")
print(f"   Channels: RGB (Red, Green, Blue)")

# =================== STEP 4: PREPROCESSING DATA ===================
print("\n🔧 STEP 4: Preprocessing Data")

# Normalisasi pixel values dari 0-255 ke 0-1
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

# Convert labels ke categorical
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print(f"✅ Preprocessing selesai:")
print(f"   X_train shape: {X_train_norm.shape}")
print(f"   y_train shape: {y_train_cat.shape}")
print(f"   Pixel range: {X_train_norm.min():.1f} - {X_train_norm.max():.1f}")

# Tampilkan contoh sebelum dan sesudah normalisasi
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(X_train[0])
axes[0].set_title(f'Original (0-255)\nPixel range: {X_train[0].min()}-{X_train[0].max()}')
axes[0].axis('off')

axes[1].imshow(X_train_norm[0])
axes[1].set_title(f'Normalized (0-1)\nPixel range: {X_train_norm[0].min():.2f}-{X_train_norm[0].max():.2f}')
axes[1].axis('off')

plt.suptitle('Normalisasi Data')
plt.tight_layout()
plt.show()

# =================== STEP 5: MEMBANGUN CNN ARCHITECTURE ===================
print("\n🧠 STEP 5: Membangun Convolutional Neural Network")

print("""
🏗️ ARSITEKTUR CNN YANG AKAN DIBUAT:
   Input: (32, 32, 3) - RGB image
   
   Conv Block 1:
   ├── Conv2D(32) + ReLU
   ├── Conv2D(32) + ReLU
   └── MaxPooling2D
   
   Conv Block 2:
   ├── Conv2D(64) + ReLU
   ├── Conv2D(64) + ReLU
   └── MaxPooling2D
   
   Conv Block 3:
   ├── Conv2D(128) + ReLU
   └── MaxPooling2D
   
   Classifier:
   ├── Flatten
   ├── Dense(512) + ReLU + Dropout
   └── Dense(10) + Softmax
""")

# Buat CNN model
model = keras.Sequential([
    # Conv Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name='conv1_1'),
    layers.Conv2D(32, (3, 3), activation='relu', name='conv1_2'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    
    # Conv Block 2
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2_1'),
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2_2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    
    # Conv Block 3
    layers.Conv2D(128, (3, 3), activation='relu', name='conv3_1'),
    layers.MaxPooling2D((2, 2), name='pool3'),
    
    # Classifier
    layers.Flatten(name='flatten'),
    layers.Dense(512, activation='relu', name='dense1'),
    layers.Dropout(0.5, name='dropout'),
    layers.Dense(10, activation='softmax', name='output')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Tampilkan summary
print("\n📋 Model Architecture:")
model.summary()

# Visualisasi arsitektur
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
plt.show()

# =================== STEP 6: DEMO CONVOLUTIONAL OPERATION ===================
print("\n🔍 STEP 6: Demo Operasi Konvolusi")

# Ambil satu gambar untuk demo
demo_image = X_train_norm[0:1]  # Shape: (1, 32, 32, 3)

# Ekstrak feature maps dari layer pertama
demo_model = keras.Model(inputs=model.input, outputs=model.get_layer('conv1_1').output)
feature_maps = demo_model.predict(demo_image, verbose=0)

print(f"Input shape: {demo_image.shape}")
print(f"Feature maps shape: {feature_maps.shape}")
print(f"✅ Conv2D menghasilkan {feature_maps.shape[-1]} feature maps")

# Visualisasi beberapa feature maps
plt.figure(figsize=(15, 8))

# Gambar asli
plt.subplot(3, 6, 1)
plt.imshow(demo_image[0])
plt.title('Original Image')
plt.axis('off')

# Tampilkan 8 feature maps pertama
for i in range(8):
    plt.subplot(3, 6, i+2)
    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
    plt.title(f'Feature Map {i+1}')
    plt.axis('off')

# Tampilkan beberapa filter weights
conv_weights = model.get_layer('conv1_1').get_weights()[0]  # Shape: (3, 3, 3, 32)

for i in range(6):
    plt.subplot(3, 6, i+11)
    # Visualisasi filter (ambil channel RGB pertama)
    filter_img = conv_weights[:, :, 0, i]
    plt.imshow(filter_img, cmap='RdBu')
    plt.title(f'Filter {i+1}')
    plt.axis('off')

plt.suptitle('Convolutional Operations: Input → Feature Maps → Filters')
plt.tight_layout()
plt.show()

print("💡 Feature maps menunjukkan area yang 'diaktivasi' oleh filter tertentu")

# =================== STEP 7: TRAINING CNN ===================
print("\n🔄 STEP 7: Training CNN Model")
print("Training CNN lebih lambat dari regular NN karena operasi konvolusi")

# Callback untuk monitoring
class CNNCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"   Epoch {epoch+1}: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}, "
              f"val_loss={logs['val_loss']:.4f}, val_acc={logs['val_accuracy']:.4f}")

# Training dengan data yang lebih kecil untuk demo (agar cepat)
print("🔥 Memulai training (menggunakan subset data untuk demo)...")

# Gunakan subset untuk demo cepat
subset_size = 5000
X_train_subset = X_train_norm[:subset_size]
y_train_subset = y_train_cat[:subset_size]

history = model.fit(
    X_train_subset, y_train_subset,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
    callbacks=[CNNCallback()]
)

print("✅ Training selesai!")

# =================== STEP 8: EVALUASI MODEL ===================
print("\n📊 STEP 8: Evaluasi Performa CNN")

# Evaluasi pada subset test data
test_subset_size = 1000
X_test_subset = X_test_norm[:test_subset_size]
y_test_subset = y_test_cat[:test_subset_size]
y_test_labels = y_test[:test_subset_size]

test_loss, test_accuracy = model.evaluate(X_test_subset, y_test_subset, verbose=0)

print(f"🎯 Hasil Evaluasi (pada subset test):")
print(f"   Loss: {test_loss:.4f}")
print(f"   Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Prediksi
predictions = model.predict(X_test_subset, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# Akurasi per class
print(f"\n📈 Akurasi per kategori:")
for i, class_name in enumerate(class_names):
    mask = y_test_labels.flatten() == i
    if np.sum(mask) > 0:
        class_acc = np.mean(predicted_classes[mask] == y_test_labels.flatten()[mask])
        print(f"   {class_name}: {class_acc:.3f} ({class_acc*100:.1f}%)")

# =================== STEP 9: VISUALISASI HASIL ===================
print("\n📈 STEP 9: Visualisasi Training & Prediksi")

plt.figure(figsize=(15, 10))

# Training history
plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion Matrix (simplified)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_labels.flatten(), predicted_classes)

plt.subplot(2, 3, 3)
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, [name[:4] for name in class_names], rotation=45)
plt.yticks(tick_marks, [name[:4] for name in class_names])

# Contoh prediksi benar dan salah
correct_indices = np.where(predicted_classes == y_test_labels.flatten())[0]
incorrect_indices = np.where(predicted_classes != y_test_labels.flatten())[0]

# Prediksi benar
if len(correct_indices) > 0:
    idx = correct_indices[0]
    plt.subplot(2, 3, 4)
    plt.imshow(X_test_subset[idx])
    plt.title(f'✅ Benar\nPrediksi: {class_names[predicted_classes[idx]]}')
    plt.axis('off')

# Prediksi salah
if len(incorrect_indices) > 0:
    idx = incorrect_indices[0]
    plt.subplot(2, 3, 5)
    plt.imshow(X_test_subset[idx])
    actual = y_test_labels.flatten()[idx]
    predicted = predicted_classes[idx]
    plt.title(f'❌ Salah\nAsli: {class_names[actual]}\nPrediksi: {class_names[predicted]}')
    plt.axis('off')

# Confidence distribution
plt.subplot(2, 3, 6)
max_confidences = np.max(predictions, axis=1)
plt.hist(max_confidences, bins=20, alpha=0.7)
plt.title('Distribution of Max Confidence')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =================== STEP 10: TRANSFER LEARNING DEMO ===================
print("\n🔄 STEP 10: Demo Transfer Learning")
print("menggunakan model pre-trained (VGG16) untuk performa lebih baik")

# Load VGG16 pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False  # Freeze weights

# Tambahkan classifier baru
transfer_model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

transfer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("📋 Transfer Learning Model:")
print(f"   Base model: VGG16 (pre-trained on ImageNet)")
print(f"   Trainable params: {sum(p.numel() for p in transfer_model.parameters() if p.requires_grad) if hasattr(transfer_model, 'parameters') else 'N/A'}")

# Quick training (1 epoch untuk demo)
print("🔄 Quick training transfer learning model...")
transfer_history = transfer_model.fit(
    X_train_subset[:1000], y_train_subset[:1000],
    epochs=2,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# =================== STEP 11: MODE INTERAKTIF ===================
print("\n🎮 STEP 11: Mode Interaktif - Test Computer Vision!")

def predict_image(model, image_index, model_name="CNN"):
    """Fungsi untuk prediksi gambar"""
    image = X_test_subset[image_index:image_index+1]
    
    # Prediksi
    prediction = model.predict(image, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    return predicted_class, confidence, prediction[0]

try:
    print("\n🎯 Test Computer Vision Model!")
    
    while True:
        print("\n" + "="*50)
        
        # Pilih model
        model_choice = input("Pilih model: (c)nn, (t)ransfer learning, atau (q)uit: ").lower()
        
        if model_choice == 'q':
            break
        elif model_choice == 'c':
            selected_model = model
            model_name = "CNN"
        elif model_choice == 't':
            selected_model = transfer_model
            model_name = "Transfer Learning"
        else:
            print("❌ Pilihan tidak valid")
            continue
        
        # Pilih gambar
        choice = input("Pilih: (r)andom, (i)ndex tertentu: ").lower()
        
        if choice == 'r':
            idx = np.random.randint(0, len(X_test_subset))
        elif choice == 'i':
            try:
                idx = int(input(f"Masukkan index (0-{len(X_test_subset)-1}): "))
                if not 0 <= idx < len(X_test_subset):
                    print("❌ Index tidak valid")
                    continue
            except ValueError:
                print("❌ Masukkan angka yang valid")
                continue
        else:
            print("❌ Pilihan tidak valid")
            continue
        
        # Prediksi
        predicted_class, confidence, all_probs = predict_image(selected_model, idx, model_name)
        actual_class = y_test_labels.flatten()[idx]
        
        # Tampilkan hasil
        plt.figure(figsize=(12, 5))
        
        # Gambar
        plt.subplot(1, 2, 1)
        plt.imshow(X_test_subset[idx])
        plt.title(f'Test Image #{idx}')
        plt.axis('off')
        
        # Probability distribution
        plt.subplot(1, 2, 2)
        colors = ['green' if i == predicted_class else 'skyblue' for i in range(10)]
        bars = plt.bar(range(10), all_probs, color=colors)
        plt.title(f'{model_name} - Probability Distribution')
        plt.xlabel('Category')
        plt.ylabel('Probability')
        plt.xticks(range(10), [name[:4] for name in class_names], rotation=45)
        
        # Highlight prediksi tertinggi
        bars[predicted_class].set_color('red')
        
        plt.tight_layout()
        plt.show()
        
        # Print hasil
        print(f"\n🎯 HASIL PREDIKSI ({model_name}):")
        print(f"   Kategori aktual: {class_names[actual_class]}")
        print(f"   Prediksi model: {class_names[predicted_class]}")
        print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        
        if predicted_class == actual_class:
            print("   ✅ PREDIKSI BENAR!")
        else:
            print("   ❌ PREDIKSI SALAH!")
            
        # Top 3 predictions
        top3_indices = np.argsort(all_probs)[-3:][::-1]
        print(f"\n📊 Top 3 Prediksi:")
        for i, class_idx in enumerate(top3_indices):
            print(f"   {i+1}. {class_names[class_idx]}: {all_probs[class_idx]:.3f} ({all_probs[class_idx]*100:.1f}%)")
            
        # Analisis kesulitan
        if confidence < 0.5:
            print(f"\n💡 Model tidak yakin - gambar mungkin ambigu")
        elif confidence > 0.9:
            print(f"\n💡 Model sangat yakin dengan prediksi")
            
except KeyboardInterrupt:
    print("\n👋 Program dihentikan.")

print("\n" + "="*60)
print("🎓 SELAMAT! Anda telah menyelesaikan program Computer Vision!")
print("🔥 Konsep yang telah Anda pelajari:")
print("   ✓ Computer Vision dan CNN fundamentals")
print("   ✓ Convolutional layers dan Feature Maps")
print("   ✓ Pooling layers dan dimensionality reduction")
print("   ✓ Image preprocessing dan normalisasi")
print("   ✓ Transfer Learning dengan pre-trained models")
print("   ✓ Model evaluation untuk klasifikasi gambar")
print("   ✓ Visualisasi filters dan feature detection")
print("\n🎉 Selamat! Anda telah menyelesaikan semua program AI fundamental!")
print("📚 Tingkatkan kemampuan dengan:")
print("   • Eksperimen dengan dataset lain")
print("   • Coba arsitektur CNN yang berbeda")
print("   • Pelajari object detection dan segmentation")
print("   • Eksplorasi Generative AI (GAN, VAE)")
print("="*60)