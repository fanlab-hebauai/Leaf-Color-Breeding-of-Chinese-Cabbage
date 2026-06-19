import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from Attation import ca_block, cbam_block, se_block, eca_block

def unet(pretrained_weights=None, input_size=(256, 256, 5), classNum=2, learning_rate=1e-5):
    inputs = Input(input_size)
    
    def conv_block(input_tensor, num_filters, name):
        x = BatchNormalization()(Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_tensor))
        x = BatchNormalization()(Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x))
        return x

    def add_attention_blocks(x, name):
        x = se_block(x, ratio=16, name=f"{name}_se_block")
        x = ca_block(x, ratio=8, name=f"{name}_ca_block")
        return x

    conv1 = conv_block(inputs, 8, "block1")
    conv1 = add_attention_blocks(conv1, "block1")
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block(pool1, 16, "block2")
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block(pool2, 32, "block3")
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block(pool3, 64, "block4")
    conv4 = add_attention_blocks(conv4, "block4")
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = conv_block(pool4, 128, "block5")
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = conv_block(merge6, 64, "block6")
    conv6 = add_attention_blocks(conv6, "block6_up")

    up7 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = conv_block(merge7, 32, "block7")
    conv7 = add_attention_blocks(conv7, "block7_up")

    up8 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = conv_block(merge8, 16, "block8")
    conv8 = add_attention_blocks(conv8, "block8_up")

    up9 = Conv2D(8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = conv_block(merge9, 8, "block9")
    conv9 = add_attention_blocks(conv9, "block9_up")

    conv10 = Conv2D(classNum, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# if __name__ == "__main__":
#     model = unet()
#     model.summary()
