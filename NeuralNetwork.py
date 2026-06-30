from tensorflow.keras import layers, models, regularizers
import tensorflow as tf

class SelfAttentionBlock(layers.Layer):
    def __init__(self, filters, reduction=16, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.reduction = reduction
        # Уменьшаем размерность для attention
        self.query_conv = layers.Conv2D(filters // reduction, (1, 1), padding='same')
        self.key_conv = layers.Conv2D(filters // reduction, (1, 1), padding='same')
        self.value_conv = layers.Conv2D(filters, (1, 1), padding='same')
        # Добавляем spatial reduction
        self.pool = layers.MaxPooling2D((2, 2))
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        # Уменьшаем пространственную размерность для attention
        pooled = self.pool(inputs)
        h_pool = tf.shape(pooled)[1]
        w_pool = tf.shape(pooled)[2]
        
        q = self.query_conv(pooled)
        k = self.key_conv(pooled)
        v = self.value_conv(pooled)
        
        # Reshape с уменьшенной размерностью
        q = tf.reshape(q, [batch_size, -1, self.filters // self.reduction])
        k = tf.reshape(k, [batch_size, -1, self.filters // self.reduction])
        v = tf.reshape(v, [batch_size, -1, self.filters])
        
        # Восстанавливаем attention scores
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.filters // self.reduction, tf.float32))
        attention_scores = tf.nn.softmax(attention_scores)
        
        attention_out = tf.matmul(attention_scores, v)
        attention_out = tf.reshape(attention_out, [batch_size, h_pool, w_pool, self.filters])
        
        attention_out = tf.image.resize(attention_out, [height, width])
        
        return inputs + attention_out
def createCNN(input_shape=(224, 224, 3), num_classes=7):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)
    
    x = layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(512, (3, 3), dilation_rate=2, activation='leaky_relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), dilation_rate=2, activation='leaky_relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    spatial_att_1 = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
    spatial_att_3 = layers.Conv2D(1, (5, 5), padding='same', activation='sigmoid')(x)
    spatial_att = layers.Average()([spatial_att_1, spatial_att_3])
    x_spatial = layers.multiply([x, spatial_att])
    
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(x.shape[-1] // 8, activation='leaky_relu')(se)
    se = layers.Dense(x.shape[-1], activation='sigmoid')(se)
    se = layers.Reshape((1, 1, x.shape[-1]))(se)
    x_se = layers.multiply([x, se])
    
    x_self_att = SelfAttentionBlock(512)(x)
    
    x_combined = layers.Concatenate()([x_spatial, x_se, x_self_att])
    x_combined = layers.Conv2D(512, (1, 1), padding='same', activation='leaky_relu')(x_combined)
    x_combined = layers.BatchNormalization()(x_combined)
    
    x_gap = layers.GlobalAveragePooling2D()(x_combined)
    x_gmp = layers.GlobalMaxPooling2D()(x_combined)
    x_pooled = layers.Concatenate()([x_gap, x_gmp])
    x_pooled = layers.Dense(1024, activation='leaky_relu')(x_pooled)
    x_pooled = layers.Dropout(0.3)(x_pooled)
    
    x = layers.Dense(1024, activation='leaky_relu', kernel_regularizer=regularizers.l2(0.0003))(x_pooled)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, activation='leaky_relu', kernel_regularizer=regularizers.l2(0.0003))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='leaky_relu', kernel_regularizer=regularizers.l2(0.0003))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
