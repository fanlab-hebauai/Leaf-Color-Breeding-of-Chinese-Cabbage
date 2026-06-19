import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LayerNormalization

def transformer_block(inputs, num_heads=1, ff_dim=128, dropout_rate=0.1):
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1] // num_heads)(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn_output = tf.keras.layers.Dense(ff_dim, activation='relu')(out1)
    ffn_output = tf.keras.layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def unet_with_transformer(pretrained_weights=None, input_size=(256, 256, 5), classNum=2, learning_rate=1e-5):
    inputs = Input(input_size)
    conv1 = BatchNormalization()(Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs))
    conv1 = BatchNormalization()(Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = BatchNormalization()(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1))
    conv2 = BatchNormalization()(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = BatchNormalization()(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2))
    conv3 = BatchNormalization()(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = BatchNormalization()(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3))
    conv4 = BatchNormalization()(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4))
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = BatchNormalization()(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4))
    conv5 = BatchNormalization()(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5))
    drop5 = Dropout(0.5)(conv5)
    
    transformer1 = transformer_block(drop5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(transformer1))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = BatchNormalization()(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6))
    conv6 = BatchNormalization()(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6))
    
    transformer2 = transformer_block(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(transformer2))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = BatchNormalization()(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7))
    conv7 = BatchNormalization()(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7))
    
    transformer3 = transformer_block(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(transformer3))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = BatchNormalization()(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8))
    conv8 = BatchNormalization()(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8))
    
    transformer4 = transformer_block(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(transformer4))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = BatchNormalization()(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9))
    conv9 = BatchNormalization()(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9))
    
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(classNum, 1, activation='softmax')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    
    return model
