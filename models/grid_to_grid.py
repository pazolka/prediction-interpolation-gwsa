from tensorflow.keras import Model
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Lambda, Concatenate, UpSampling2D, Dropout, Input, UpSampling3D, SpatialDropout3D, Dropout
from keras.models import Model

def unet2d(input_shape, dropout_rate=0.1):

    inputs = Input(input_shape)

    # ---- Encoder ----
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    if dropout_rate: c1 = Dropout(dropout_rate, name='drop_c1')(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # Down to 12x12
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    if dropout_rate: c2 = Dropout(dropout_rate, name='drop_c2')(c2)
    p2 = MaxPooling2D((2, 2))(c2)  # Down to 6x6
    
    # ---- Bottleneck ----
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    if dropout_rate: c3 = Dropout(dropout_rate, name='drop_c32d')(c3)

    # ---- Decoder ----
    u4 = UpSampling2D((2, 2))(c3)  # Back to 12x12
    u4 = Concatenate()([u4, c2])
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(u4)
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(c4)
    
    u5 = UpSampling2D((2, 2))(c4)  # Back to 24x24
    u5 = Concatenate()([u5, c1])
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(c5)
    
    # Output layer for regression
    outputs = Conv2D(1, (1, 1), activation='linear')(c5)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def unet3d(
    input_shape,
    base=16,
    dropout_rate=0.1,
    spatial_dropout=False
):
    Drop = SpatialDropout3D if spatial_dropout else Dropout

    inputs = Input(input_shape)

    # ---- Encoder ----
    c1 = Conv3D(base, (3,3,3), padding='same', activation='relu')(inputs)
    c1 = Conv3D(base, (3,3,3), padding='same', activation='relu')(c1)
    if dropout_rate: c1 = Drop(dropout_rate, name='drop_c1')(c1)
    p1 = MaxPooling3D(pool_size=(1,2,2))(c1)

    c2 = Conv3D(base*2, (3,3,3), padding='same', activation='relu')(p1)
    c2 = Conv3D(base*2, (3,3,3), padding='same', activation='relu')(c2)
    if dropout_rate: c2 = Drop(dropout_rate, name='drop_c2')(c2)
    p2 = MaxPooling3D(pool_size=(1,2,2))(c2)

    # ---- Bottleneck ----
    b  = Conv3D(base*4, (3,3,3), padding='same', activation='relu')(p2)
    b  = Conv3D(base*4, (3,3,3), padding='same', activation='relu')(b)
    if dropout_rate: b = Drop(dropout_rate, name='drop_b')(b)

    # ---- Decoder ----
    u4 = UpSampling3D((1,2,2))(b)
    u4 = Conv3D(base*2, (3,3,3), padding='same', activation='relu')(u4)
    u4 = Concatenate()([u4, c2])
    u4 = Conv3D(base*2, (3,3,3), padding='same', activation='relu')(u4)
    u4 = Conv3D(base*2, (3,3,3), padding='same', activation='relu')(u4)

    u5 = UpSampling3D((1,2,2))(u4)
    u5 = Conv3D(base, (3,3,3), padding='same', activation='relu')(u5)
    u5 = Concatenate()([u5, c1])
    u5 = Conv3D(base, (3,3,3), padding='same', activation='relu')(u5)
    u5 = Conv3D(base, (3,3,3), padding='same', activation='relu')(u5)

    outputs = Conv3D(1, (1,1,1), padding='same', activation='linear')(u5)
    output = Lambda(lambda t: t[:, -1], name='last')(outputs)

    return Model(inputs, output, name='UNet3D_reg')
