"""HIKNet: the small 1D + late 2D convolutional network from the paper.

Input  : (199, 6)  six channels over 199 time steps (standardized)
Layers : Conv1D(150, k=15) -> MaxPool(2) -> Dropout(0.4)
         Conv1D(150, k=15) -> MaxPool(2) -> Dropout(0.4)
         reshape to (39, 150, 1) -> Conv2D(150, (3, 15), stride (3, 1))
         -> GlobalAveragePooling2D -> Dropout(0.4) -> Dense(1, sigmoid)

Shapes, no padding: 199 -> 185 -> 92 -> 78 -> 39 -> (13, 136, 150) -> 150 -> 1.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_hiknet(n_timesteps=199, n_channels=6, filters=150, kernel=15, dropout=0.4):
    inputs = keras.Input(shape=(n_timesteps, n_channels))

    x = layers.Conv1D(filters, kernel, activation="relu")(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(filters, kernel, activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Reshape((x.shape[1], x.shape[2], 1))(x)
    x = layers.Conv2D(filters, (3, kernel), strides=(3, 1), activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="HIKNet")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    build_hiknet().summary()
