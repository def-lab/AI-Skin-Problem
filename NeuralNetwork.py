def createCNN(input_shape=(224, 224, 3), num_classes=7):
    inputs = layers.Input(shape=input_shape)

    x = layers.Rescaling(1. / 127.5, offset=-1)(inputs)

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

    def self_attention_block(input_tensor, filters):
        q = layers.Conv2D(filters // 8, (1, 1), padding='same')(input_tensor)
        k = layers.Conv2D(filters // 8, (1, 1), padding='same')(input_tensor)
        v = layers.Conv2D(filters, (1, 1), padding='same')(input_tensor)

        shape = tf.shape(q)
        q = layers.Reshape((shape[1] * shape[2], shape[3]))(q)
        k = layers.Reshape((shape[1] * shape[2], shape[3]))(k)
        v = layers.Reshape((shape[1] * shape[2], shape[3]))(v)

        attention_scores = layers.Dot(axes=[2, 2])([q, k])
        attention_scores = layers.Activation('softmax')(attention_scores)

        attention_out = layers.Dot(axes=[2, 1])([attention_scores, v])
        attention_out = layers.Reshape((shape[1], shape[2], filters))(attention_out)

        return layers.Add()([input_tensor, attention_out])

    x_self_att = self_attention_block(x, 512)

    x_combined = layers.Concatenate()([x_spatial, x_se, x_self_att])
    x_combined = layers.Conv2D(512, (1, 1), padding='same', activation='leaky_relu')(x_combined)
    x_combined = layers.BatchNormalization()(x_combined)

    x_gap = layers.GlobalAveragePooling2D()(x_combined)
    x_gmp = layers.GlobalMaxPooling2D()(x_combined)
    x_pooled = layers.Concatenate()([x_gap, x_gmp])
    x_pooled = layers.Dense(1024, activation='leaky_relu')(x_pooled)
    x_pooled = layers.Dropout(0.3)(x_pooled)

    x = layers.Dense(1024, activation='leaky_relu', kernel_regularizer=regularizers.l2(0.001))(x_pooled)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512, activation='leaky_relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation='leaky_relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model