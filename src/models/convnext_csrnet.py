from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.models import Model

def conv_block(x, f):
    for _ in range(2):
        x = Conv2D(f, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x

def build_convnext_crowd(input_shape=(512, 512, 3)):
    inputs = Input(shape=input_shape)
    backbone = ConvNeXtTiny(include_top=False, include_preprocessing=True,
                            weights="imagenet", input_tensor=inputs)
    backbone.trainable = False

    x = conv_block(backbone.output, 256)
    for f in [128, 64, 32, 16]:
        x = UpSampling2D(2)(x)
        x = conv_block(x, f)
    x = UpSampling2D(2)(x)
    return Model(inputs, Conv2D(1, 1, activation="relu")(x))
