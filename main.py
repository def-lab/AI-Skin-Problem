
from livelossplot import PlotLossesKeras
from NeuralNetwork import *
from dataForCNN import load_dataset
import tensorboard
from importData import *
import keras
CNN = createCNN()

CNN.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])
train_df,val_df = formdata()
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
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")
history = CNN.fit(
    train_ds,
    steps_per_epoch=len(train_df)//32,
    epochs=20,
    callbacks=[tensorboard_callback],
    validation_data=val_ds,
    validation_steps=len(val_df)//32,
    verbose = 1
)
# Посмотреть метрики
print("\nФинальные метрики:")
print(f"Train Accuracy: {history.history['sparse_categorical_accuracy'][-1]:.4f}")
print(f"Val Accuracy: {history.history['val_sparse_categorical_accuracy'][-1]:.4f}")
print(f"Train Loss: {history.history['loss'][-1]:.4f}")
print(f"Val Loss: {history.history['val_loss'][-1]:.4f}")
CNN.save_weights("weights.h5")