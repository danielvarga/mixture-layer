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


GAUSS_PARAM_COUNT = 5

class MixtureLayer(Layer):
    # learn_variance==True means that a diagonal covariance matrix is learned.
    # if learn_variance, then variance arg is interpreted as maximum possible value,
    # if not, then it's the only possible value.
    def __init__(self, size, learn_variance=True, variance=1.0/200, maxpooling=True, **kwargs):
        self.output_dim = 2
        self.size = size
        self.learn_variance = learn_variance
        self.variance = variance
        self.maxpooling = maxpooling
        super(MixtureLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[2] == GAUSS_PARAM_COUNT # x, y, xv, yv, density
        self.k = input_shape[1]
        super(MixtureLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inp, mask=None):
        k = self.k
        size = self.size
        assert GAUSS_PARAM_COUNT == 5
        xs = inp[:, :, 0]
        ys = inp[:, :, 1]
        xv = inp[:, :, 2]
        yv = inp[:, :, 3]
        densities = inp[:, :, 4]

        xi = tf.linspace(0.0, 1.0, size)
        xi = tf.reshape(xi, [1, 1, 1, -1])
        xi = tf.tile(xi, [1, k, size, 1])
        # -> xi.shape==(1, k, size, size), xi[0][0] has #size different cols, each col has #size identical numbers in it.
        yi = tf.transpose(xi, [0, 1, 3, 2])
        
        def add_two_dims(t):
            return K.expand_dims(K.expand_dims(t))
        xse = add_two_dims(xs)
        yse = add_two_dims(ys)
        xve = add_two_dims(xv)
        yve = add_two_dims(yv)
        de  = add_two_dims(densities)

        if self.learn_variance:
            # learned diagonal covariance. SD never bigger than 0.07:
            error = (xi - xse) ** 2 / (xve * self.variance) + (yi - yse) ** 2 / (yve * self.variance)
        else:
            # 0.0005 is a nice little dot useful for learning MNIST
            error = (xi - xse) ** 2 / self.variance + (yi - yse) ** 2 / self.variance
        error /= 2

        # avgpooling is better for reconstruction (if negative ds are allowed),
        # val_loss: 0.0068, but way-way worse for interpolation, it looks like a smoke monster.
        # Note that fixed variance maxpooling will never generalize beyond MNIST.
        if self.maxpooling:
            out = K.max(de * K.exp(-error), axis=1)
        else:
            out = K.sum((2 * de - 1) * K.exp(-error), axis=1)
        return out

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


def interpolate(sample_a, sample_b, encoder, decoder, frame_count, output_image_size):
    latent = encoder.predict(np.array([sample_a, sample_b]).reshape(2, -1))
    latent_a, latent_b = latent

    latents = []
    for t in np.linspace(0.0, 1.0, frame_count):
        l = (1-t) * latent_a + t * latent_b
        latents.append(l)
    latents = np.array(latents)
    interp = decoder.predict(latents)
    interp = interp.reshape(frame_count, output_image_size, output_image_size)
    return interp


def saveModel(model, filePrefix):
    jsonFile = filePrefix + ".json"
    weightFile = filePrefix + ".h5"
    with open(filePrefix + ".json", "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(weightFile)
    print "Saved model to files {}, {}".format(jsonFile, weightFile)


def load_mnist():
    image_size = 28
    nb_features = image_size * image_size

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
    return image_size, (X_train, y_train), (X_test, y_test)


def load_celebabw():
    image_size = 64
    nb_features = image_size * image_size

    celeba = np.load("/home/csadrian/datasets/celeba6472.npy").astype(np.float32)
    celeba = celeba[:, 4:-4, :]

    X_train, X_test = celeba[:100000], celeba[100000:110000]
    X_train = X_train.reshape(len(X_train), -1)
    X_test  = X_test .reshape(len(X_test ), -1)

    return image_size, (X_train, np.zeros(len(X_train), dtype=np.int32)), (X_test, np.zeros(len(X_test), dtype=np.int32))


def test_learn():
    data_source = "celebabw"

    assert data_source in ("mnist", "celebabw")
    if data_source == "mnist":
        image_size, (X_train, y_train), (X_test, y_test) = load_mnist()
    elif data_source == "celebabw":
        image_size, (X_train, y_train), (X_test, y_test) = load_celebabw()

    nb_features = image_size * image_size

    batch_size = 128
    nb_epoch = 1
    k = 300
    nonlinearity = 'relu'
    intermediate_layer_size = 1000

    if data_source == "mnist":
        learn_variance = False
        variance = 0.0005
    elif data_source == "celebabw":
        learn_variance = True
        variance = 1.0/200 # Interpreted as maximum allowed variance 7% of image size.
    maxpooling = True
    
    mixture_layer = MixtureLayer(image_size, learn_variance=learn_variance, variance=variance, maxpooling=maxpooling)

    inputs = Input(shape=(nb_features,))
    net = inputs
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(k * GAUSS_PARAM_COUNT, activation='sigmoid')(net)
    gaussians = Reshape((k, GAUSS_PARAM_COUNT))(net)
    net = mixture_layer(gaussians)
    net = Reshape((nb_features,))(net)
    model = Model(input=inputs, output=net)

    model.summary()

    model.compile(loss='mse', optimizer=Adam())

    history = model.fit(X_train, X_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, X_test))

    saveModel(model, "model")

    # n = 400
    # displaySet(X_train[:n].reshape(n, image_size, image_size), n, model, "ae-train", flatten_input=True)
    # displaySet(X_test [:n].reshape(n, image_size, image_size), n, model, "ae-test",  flatten_input=True)

    encoder = Model(input=inputs, output=gaussians)
    encoder.compile(loss='mse', optimizer=SGD())

    input_gaussians = Input(shape=(k, GAUSS_PARAM_COUNT))
    output_image_size = image_size * 2 # We can increase the resolution
    mixture_layer_2 = MixtureLayer(output_image_size, learn_variance=learn_variance, variance=variance, maxpooling=maxpooling)
    decoder_layer = mixture_layer_2(input_gaussians)
    decoder_layer = Reshape((output_image_size*output_image_size,))(decoder_layer)
    decoder = Model(input=input_gaussians, output=decoder_layer)
    decoder.compile(loss='mse', optimizer=SGD())

    frame_count = 30

    # interp = interpolate(X_train[31], X_train[43], encoder, decoder, frame_count, output_image_size)
    # plotImages(interp, 10, 10, "ae-interp")

    animation = []

    targets = []
    def collect(target_digit, anim_phases):
        i = 0
        j = 0
        while j < anim_phases:
            if y_train[i] == target_digit:
                targets.append(i)
                j += 1
            i += 1

    if data_source == "mnist":
        anim_phases = 10
        collect(3, anim_phases)
        collect(5, anim_phases)
        targets += range(anim_phases)
    elif data_source == "celebabw":
        anim_phases = 30
        # Not using supervised data
        targets += range(anim_phases)
    print "Animation phase count %d" % len(targets)

    for i in range(len(targets)-1):
        interp = interpolate(X_train[targets[i]], X_train[targets[i+1]], encoder, decoder, frame_count, output_image_size)
        animation.extend(interp[:-1])

    print "Creating frames of animation"
    for i, frame_i in enumerate(animation):
        img = Image.fromarray((255 * np.clip(frame_i, 0.0, 1.0)).astype(dtype='uint8'), mode="L")
        img.save("gif/%03d.gif" % i)

test_learn()
