from tensorflow.keras import layers, models, regularizers

def createCNN(input_shape=(224, 224, 3), num_classes=7):
    model = models.Sequential([
        # Блок 1 - больше фильтров для лучшего захвата признаков
        layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Блок 2
        layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Блок 3
        layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Блок 4
        layers.Conv2D(512, (3, 3), activation='leaky_relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='leaky_relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Глобальный пулинг
        layers.GlobalAveragePooling2D(),

        # Полносвязные слои
        layers.Dense(512, activation='leaky_relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='leaky_relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='leaky_relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model



