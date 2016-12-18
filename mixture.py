import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

from PIL import Image


class MixtureLayer(Layer):
    def __init__(self, size, **kwargs):
        self.output_dim = 2
        self.size = size
        super(MixtureLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[2] == 3 # x, y, density
        self.k = input_shape[1]
        super(MixtureLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inp, mask=None):
        k = self.k
        size = self.size
        xs = inp[:, :, 0]
        ys = inp[:, :, 1]
        densities = inp[:, :, 2]
        xi = tf.linspace(0.0, 1.0, size)
        xi = tf.reshape(xi, [1, 1, 1, -1])
        xi = tf.tile(xi, [1, k, size, 1])
        # -> xi.shape==(1, k, size, size), xi[0][0] has #size different cols, each col has #size identical numbers in it.
        yi = tf.transpose(xi, [0, 1, 3, 2])
        
        xse = K.expand_dims(xs)
        xse = K.expand_dims(xse)
        yse = K.expand_dims(ys)
        yse = K.expand_dims(yse)
        
        variance = 0.01
        error = (xi - xse) ** 2 + (yi - yse) ** 2
        error /= 2 * variance
        # TODO densities ignored yet
        return K.sum(K.exp(-error), axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


def main():
    inp = np.array([[0.8, 0.8, 0.5], [0.3, 0.2, 0.5]]) # each row specifies a gaussian, as [x, y, density]
    size = 100
    inputs = Input(shape=inp.shape)
    net = MixtureLayer(size)(inputs)
    model = Model(input=inputs, output=net)
    out = model.predict([np.expand_dims(inp, 0)])
    out = out[0]
    print out.shape
    out *= 255.0 / np.max(out)

    img = Image.fromarray(out.astype(dtype='uint8'), mode="L")
    img.save("vis.png")


main()



def learn():
    # The data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # flatten the 28x28 images to arrays of length 28*28:
    X_train = X_train.reshape(60000, nb_features)
    X_test = X_test.reshape(10000, nb_features)

    # convert brightness values from bytes to floats between 0 and 1:
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    image_size = 28
    nb_features = image_size * image_size
    layer_size = 30
    nonlinearity = 'relu'
    inputs = Input(shape=(nb_features,))
    net = Dense(layer_size, activation=nonlinearity)(inputs)
