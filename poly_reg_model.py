from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import Reshape

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense,Flatten,Dropout
import tensorflow as tf

def Conv2D_BN_Leaky(input_tensor, *args):
    output_tensor = Conv2D(*args, 
                           padding='same',
                           kernel_initializer='he_normal')(input_tensor)
    output_tensor = BN()(output_tensor)
    output_tensor = LeakyReLU(alpha=0.1)(output_tensor)
    return output_tensor

def backbone(x):
    conv1 = Conv2D_BN_Leaky(x, 64, 7, 2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D_BN_Leaky(pool1, 192, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D_BN_Leaky(pool2, 128, 1)
    conv3 = Conv2D_BN_Leaky(conv3, 256, 3)
    conv3 = Conv2D_BN_Leaky(conv3, 256, 1)
    conv3 = Conv2D_BN_Leaky(conv3, 512, 3)
    pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)

    conv4 = pool3
    for _ in range(4):
        conv4 = Conv2D_BN_Leaky(conv4, 256, 1)
        conv4 = Conv2D_BN_Leaky(conv4, 512, 3)
    conv4 = Conv2D_BN_Leaky(conv4, 512, 1)
    conv4 = Conv2D_BN_Leaky(conv4, 1024, 3)
    pool4 = MaxPooling2D(pool_size=(2, 1))(conv4)

    conv5 = Conv2D_BN_Leaky(pool4, 512, 1)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv5 = Conv2D_BN_Leaky(conv5, 512, 1)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3, (2,1))

    conv6 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv6 = Conv2D_BN_Leaky(conv6, 1024, 3)
    return conv6

def poly_reg(input_shape=(488, 400, 3),output_shape=(50,3)):
    inputs = Input(input_shape)
    darknet_outputs = backbone(inputs)
    outputs= Conv2D(output_shape[1],(8,1),kernel_initializer='he_normal')(darknet_outputs)

    model = Model(inputs, outputs)

    return model
