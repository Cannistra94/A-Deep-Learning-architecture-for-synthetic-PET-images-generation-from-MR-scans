def residual_block(x, filters, kernel_size=3, stride=1, dropout_rate=0.3):
    shortcut = x
    
    # Adjust the shortcut if the number of filters or stride doesn't match
    if stride != 1 or x.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dropout(dropout_rate)(x)  # Dropout after the first activation --remove in case
    
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    #x = Dropout(dropout_rate)(x)  # Dropout after the second activation
    
    return x


def resnet_unet_2d_1mm_new(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = residual_block(inputs, 64)
    c1 = residual_block(c1, 64)  # Additional block
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = residual_block(p1, 128)
    c2 = residual_block(c2, 128)  # Additional block
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = residual_block(p2, 256)
    c3 = residual_block(c3, 256)  # Additional block
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = residual_block(p3, 512)
    c4 = residual_block(c4, 512)  # Additional block
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    c5 = residual_block(p4, 1024)
    c5 = residual_block(c5, 1024)  # Additional block
    
    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.ZeroPadding2D(padding=((0, 0), (1, 0)))(u6)
    u6 = layers.concatenate([u6, c4])
    c6 = residual_block(u6, 512)
    
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.ZeroPadding2D(padding=((1, 0), (0, 0)))(u7)
    u7 = layers.concatenate([u7, c3])
    c7 = residual_block(u7, 256)
    
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(u8)
    u8 = layers.concatenate([u8, c2])
    c8 = residual_block(u8, 128)
    
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.ZeroPadding2D(padding=((0, 0), (0, 0)))(u9)
    u9 = layers.concatenate([u9, c1])
    c9 = residual_block(u9, 64)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=[combined_mae_mse_loss], metrics=[psnr_metric])
    
    return model

# Define the input shape
input_shape = (182, 218, 1)
model = resnet_unet_2d_1mm_new(input_shape)
model.summary()
