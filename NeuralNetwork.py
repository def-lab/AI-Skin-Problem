from tensorflow.keras import layers, models,regularizers

def createCNN(input_shape=(224, 224, 3), num_classes=7):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation="leaky_relu", padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(32, (3, 3), activation="leaky_relu", padding='same',kernel_regularizer = regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation="leaky_relu", padding='same',kernel_regularizer = regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), activation="leaky_relu", padding='same',kernel_regularizer = regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.35),
    
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="leaky_relu",kernel_regularizer = regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation="leaky_relu",kernel_regularizer = regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation="softmax")
    ])
    
    return model
