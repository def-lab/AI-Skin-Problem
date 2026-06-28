
from livelossplot import PlotLossesKeras
from NeuralNetwork import *
from dataForCNN import load_dataset
import tensorboard
from importData import *
import keras
CNN = createCNN()
train_ds= load_dataset(
    train_df,
    batch_size=32,
    is_training=True,
    augment=True
)
val_ds= load_dataset(
    val_df,
    batch_size=32,
    is_training=False,
    augment=False
)
initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

optimizer = keras.optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    clipnorm=1.0
)

CNN.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy']
)

callbacks = [
    tensorboard_callback,
    keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        patience=15,
        restore_best_weights=True,
        mode='max'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_sparse_categorical_accuracy',
        save_best_only=True,
        mode='max'
    )
]

history = CNN.fit(
    train_ds,
    steps_per_epoch=len(train_df)//32,
    epochs=50,
    callbacks=callbacks,
    validation_data=val_ds,
    validation_steps=len(val_df)//32,
    verbose=1
)


# Посмотреть метрики
print("\nФинальные метрики:")
print(f"Train Accuracy: {history.history['sparse_categorical_accuracy'][-1]:.4f}")
print(f"Val Accuracy: {history.history['val_sparse_categorical_accuracy'][-1]:.4f}")
print(f"Train Loss: {history.history['loss'][-1]:.4f}")
print(f"Val Loss: {history.history['val_loss'][-1]:.4f}")
CNN.save_weights("weights.h5")