import tensorflow as tf
from tensorflow import keras

# Memuat dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisasi data ke rentang [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Membangun model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Mengubah input menjadi vektor 1D
    keras.layers.Dense(128, activation='relu'),  # Lapisan tersembunyi
    keras.layers.Dense(10, activation='softmax')  # Lapisan output untuk 10 kelas (0-9)
])

# Mengompilasi model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Melatih model
model.fit(x_train, y_train, epochs=5)

# Evaluasi model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nAkurasi pengujian:', test_acc)
