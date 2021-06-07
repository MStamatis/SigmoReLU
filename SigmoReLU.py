"""
Python (Keras) implemenation of SigmoReLU activation function used in 
"SigmoReLU: An improvement activation function by combining Sigmoid and ReLU",
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Lambda
from tensorflow.keras.utils import get_custom_objects
def SigmoReLU(x):
  """
  The gradients are automatically calculated on TF2
  """
  return K.maximum(tf.keras.activations.relu(x), tf.keras.activations.sigmoid(x))
  
#usage between convolution layers
get_custom_objects().update({'SigmoReLU':
tf.keras.layers.Activation(SigmoReLU)})
conv = Conv2D(32, (5, 5))(visible)
conv_act = SigmoReLU(conv)
conv_act_batch = BatchNormalization()(conv_act)
conv_maxpool = MaxPooling2D()(conv_act_batch)
conv_dropout = Dropout(0.1)(conv_maxpool)
