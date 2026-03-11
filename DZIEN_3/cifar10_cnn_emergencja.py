import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# 1. Wczytanie danych CIFAR-10
# --------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Normalizacja
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# --------------------------------------------------
# 2. Budowa prostego modelu CNN
# --------------------------------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
    tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
    tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
    tf.keras.layers.MaxPooling2D((2, 2), name='pool3'),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --------------------------------------------------
# 3. Krótkie trenowanie
# --------------------------------------------------
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# --------------------------------------------------
# 4. Ocena modelu
# --------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Dokładność testowa: {test_acc:.4f}")

# --------------------------------------------------
# 5. Wybór przykładowego obrazu
# --------------------------------------------------
sample_index = 8
sample_image = x_test[sample_index]
sample_label = int(y_test[sample_index])

prediction = model.predict(np.expand_dims(sample_image, axis=0), verbose=0)
predicted_class = int(np.argmax(prediction))

plt.figure(figsize=(4, 4))
plt.imshow(sample_image)
plt.title(f"Wejście: true={class_names[sample_label]}, pred={class_names[predicted_class]}")
plt.axis("off")
plt.show()

# --------------------------------------------------
# 6. Model do wyciągania aktywacji pośrednich
# --------------------------------------------------
layer_names = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3']
layer_outputs = [model.get_layer(name).output for name in layer_names]

activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(np.expand_dims(sample_image, axis=0), verbose=0)

# --------------------------------------------------
# 7. Funkcja do wizualizacji map cech
# --------------------------------------------------
def plot_feature_maps(feature_maps, title, max_maps=16):
    """
    feature_maps: tensor o kształcie (1, H, W, C)
    """
    num_features = feature_maps.shape[-1]
    num_to_show = min(num_features, max_maps)

    cols = 4
    rows = int(np.ceil(num_to_show / cols))

    plt.figure(figsize=(12, 3 * rows))
    plt.suptitle(title, fontsize=16)

    for i in range(num_to_show):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.title(f"Mapa {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 8. Wyświetlenie aktywacji
# --------------------------------------------------
for layer_name, activation in zip(layer_names, activations):
    print(f"Warstwa: {layer_name}, kształt: {activation.shape}")
    plot_feature_maps(activation, f"Aktywacje warstwy: {layer_name}", max_maps=16)
