import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Activation
from keras.optimizers import SGD, Adam, RMSprop

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
        de = K.expand_dims(densities)
        de = K.expand_dims(de)
        
        variance = 0.003
        error = (xi - xse) ** 2 + (yi - yse) ** 2
        error /= 2 * variance
        # BEWARE, max not sum, mnist-specific!
        return K.max(de * K.exp(-error), axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.size, self.size)


def test_forward():
    inp = np.array([[0.8, 0.8, 0.5], [0.3, 0.2, 0.5]]) # each row specifies a gaussian, as [x, y, density]
    size = 100
    inputs = Input(shape=inp.shape)
    net = MixtureLayer(size)(inputs)
    model = Model(input=inputs, output=net)
    out = model.predict([np.expand_dims(inp, 0)])
    out = out[0]
    out = np.clip(out, 0.0, 1.0)
    out *= 255.0

    img = Image.fromarray(out.astype(dtype='uint8'), mode="L")
    img.save("vis.png")


def plotImages(data, n_x, n_y, name):
    height, width = data.shape[1:]
    height_inc = height + 1
    width_inc = width + 1
    n = len(data)
    if n > n_x*n_y: n = n_x * n_y

    mode = "L"
    image_data = np.zeros((height_inc * n_y + 1, width_inc * n_x - 1), dtype='uint8')
    for idx in xrange(n):
        x = idx % n_x
        y = idx / n_x
        sample = data[idx]
        image_data[height_inc*y:height_inc*y+height, width_inc*x:width_inc*x+width] = 255*sample.clip(0, 0.99999)
    img = Image.fromarray(image_data,mode=mode)
    fileName = name + ".png"
    print "Creating file " + fileName
    img.save(fileName)


def displaySet(imageBatch, n, generator, name, flatten_input=False):
    batchSize = imageBatch.shape[0]
    nsqrt = int(np.ceil(np.sqrt(n)))
    if flatten_input:
        net_input = imageBatch.reshape(batchSize, -1)
    else:
        net_input = imageBatch
    recons = generator.predict(net_input, batch_size=batchSize)
    recons = recons.reshape(imageBatch.shape)

    mergedSet = np.zeros(shape=[n*2] + list(imageBatch.shape[1:]))
    for i in range(n):
        mergedSet[2*i] = imageBatch[i]
        mergedSet[2*i+1] = recons[i]
    result = mergedSet.reshape([2*n] + list(imageBatch.shape[1:]))
    plotImages(result, 2*nsqrt, nsqrt, name)


def test_learn():
    image_size = 28
    nb_features = image_size * image_size
    batch_size = 512
    nb_epoch = 10
    k = 20
    nonlinearity = 'relu'
    intermediate_layer_size = 1000

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

    inputs = Input(shape=(nb_features,))
    net = inputs
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(k * 3, activation='sigmoid')(net)
    gaussians = Reshape((k, 3))(net)
    net = MixtureLayer(image_size)(gaussians)
    # net = Activation('sigmoid')(net)
    net = Reshape((nb_features,))(net)
    model = Model(input=inputs, output=net)

    model.summary()

    model.compile(loss='mse', optimizer=Adam())

    history = model.fit(X_train, X_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, X_test))

    n = 400
    displaySet(X_train[:n].reshape(n, image_size, image_size), n, model, "ae-train", flatten_input=True)
    displaySet(X_test [:n].reshape(n, image_size, image_size), n, model, "ae-test",  flatten_input=True)

    encoder = Model(input=inputs, output=gaussians)
    encoder.compile(loss='mse', optimizer=SGD())

    input_gaussians = Input(shape=(k, 3))
    output_image_size = image_size * 4 # It's cheap now!
    decoder_layer = MixtureLayer(output_image_size)(input_gaussians)
    decoder_layer = Reshape((output_image_size*output_image_size,))(decoder_layer)
    decoder = Model(input=input_gaussians, output=decoder_layer)
    decoder.compile(loss='mse', optimizer=SGD())

    sample_a = 31 # 7
    sample_b = 43 # 10

    latent = encoder.predict(X_train[[sample_a, sample_b]].reshape((2, -1)))
    latent_a, latent_b = latent

    n = 100
    latents = []
    for t in np.linspace(0.0, 1.0, n):
        l = t*latent_a + (1-t)*latent_b
        # l[6:, :] = 0
        latents.append(l)
    latents = np.array(latents)
    interp = decoder.predict(latents).reshape(n, output_image_size, output_image_size)
    plotImages(interp, 10, 10, "ae-interp")


test_learn()
