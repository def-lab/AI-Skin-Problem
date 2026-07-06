from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
def create_tl():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)

    )
    base_model.trainable = True
    for layer in base_model.layers[:50]:
        layer.trainable = False
    model=tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        layers.Dense(256,activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(7,activation='softmax')


    ])
    return model
