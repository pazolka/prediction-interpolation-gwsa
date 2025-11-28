from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, TimeDistributed, Dense, LSTM, Flatten


def cnn_lstm(sequence_length, img_height, img_width, output_size, channels, lstm1_units=128, lstm2_units=128):
    inputs = Input(shape=(sequence_length, img_height, img_width, channels))
    
    # ---- CNN part ----
    c1 = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(inputs)
    p2 = TimeDistributed(MaxPooling2D((2, 2)))(c1)
    c3 = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(p2)
    p4 = TimeDistributed(MaxPooling2D((2, 2)))(c3)
    f5 = TimeDistributed(Flatten())(p4)
    drop6 = Dropout(rate = 0.1)(f5)
    
    # ---- LSTM parts ----
    l7 = LSTM(lstm1_units, activation='relu', return_sequences=True)(drop6)
    l8 = LSTM(lstm2_units, activation='relu')(l7)

    # ---- Output layer ----
    d9 = Dense(output_size, activation='linear')(l8)
    model = Model(inputs=inputs, outputs=d9)

    return model

def cnn(img_height, img_width, channels, output_size):
    inputs = Input(shape=(img_height, img_width, channels))
    
    # ---- CNN part ----
    c1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    p2 = MaxPooling2D((2, 2))(c1)
    c3 = Conv2D(64, (3, 3), activation='relu')(p2)
    p4 = MaxPooling2D((2, 2))(c3)
    f5 = Flatten()(p4)
    drop6 = Dropout(rate = 0.2)(f5)
    
    # ---- Output layer ----
    d9 = Dense(output_size, activation='linear')(drop6)
    model = Model(inputs=inputs, outputs=d9)

    return model
