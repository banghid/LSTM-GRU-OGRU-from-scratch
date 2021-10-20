import tensorflow.compat.v2 as tf

import uuid
from tensorflow.python.eager.context import get_device_name
from keras import activations
from keras import backend
from keras.engine.input_spec import InputSpec
from keras.layers import recurrent
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

