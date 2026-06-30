
from livelossplot import PlotLossesKeras
from NeuralNetwork import *
from dataForCNN import load_dataset
import tensorboard
from importData import *
import keras
import tensorflow as tf

class GPU_cleaner(keras.callbacks.Callback):
	def on_epoch_end(self,epoch,logs = None):
		tf.keras.backend.clear_session()
CNN = createCNN()

train_df,val_df = formdata()
train_ds= load_dataset(
    train_df,
    batch_size=8,
    is_training=True,
    augment=True
)
val_ds= load_dataset(
    val_df,
    batch_size=8,
    is_training=False,
    augment=False
)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

initial_learning_rate = 0.005
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps = (len(train_df)//16)*10,
    decay_rate=0.95,
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
        './models/best_model.weights.h5',
        monitor='val_sparse_categorical_accuracy',
	save_weights_only = True,
        save_best_only=True,
        mode='max'
    ),
    GPU_cleaner()
]

history = CNN.fit(
    train_ds,
    steps_per_epoch=len(train_df)//8,
    epochs=50,
    callbacks=callbacks,
    validation_data=val_ds,
    validation_steps=len(val_df)//8,
    verbose=1
)


# Посмотреть метрики
print("\nФинальные метрики:")
print(f"Train Accuracy: {history.history['sparse_categorical_accuracy'][-1]:.4f}")
print(f"Val Accuracy: {history.history['val_sparse_categorical_accuracy'][-1]:.4f}")
print(f"Train Loss: {history.history['loss'][-1]:.4f}")
print(f"Val Loss: {history.history['val_loss'][-1]:.4f}")
CNN.save_weights("weights.h5")
