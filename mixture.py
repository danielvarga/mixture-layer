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


import data


def get_param_count(learn_variance, learn_density):
    gauss_param_count = 5
    if not learn_variance:
	gauss_param_count -= 2
    if not learn_density:
	gauss_param_count -= 1
    return gauss_param_count

class MixtureLayer(Layer):
    def __init__(self, sizeX, sizeY, learn_variance=True, learn_density=False, variance=1.0/200, maxpooling=True, **kwargs):
        self.output_dim = 2
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.learn_variance = learn_variance
        self.learn_density = learn_density
        self.variance = variance
        self.maxpooling = maxpooling

        self.xs_index = 0
        self.ys_index = 1
        if learn_variance:
            self.xv_index = 2
            self.yv_index = 3
            if learn_density:
                self.densities_index = 4
        else:
            if learn_density:
                self.densities_index = 2

        super(MixtureLayer, self).__init__(**kwargs)


    # input_shape = (batch, channels, dots, GAUSS_PARAM_COUNT)
    def build(self, input_shape):
        assert len(input_shape) == 4
        #        assert input_shape[3] == self.GAUSS_PARAM_COUNT # x, y, xv, yv, density but the last three could be missing!!!
        self.k = input_shape[2] # number of dots to place on each channel
        super(MixtureLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inp, mask=None):
        k = self.k
        sizeX = self.sizeX
        sizeY = self.sizeY

        def add_two_dims(t):
            return K.expand_dims(K.expand_dims(t))

        xs = inp[:, :, :, self.xs_index]
        ys = inp[:, :, :, self.ys_index]
        xse = add_two_dims(xs)
        yse = add_two_dims(ys)
        if self.learn_variance:
            xv = inp[:, :, :, self.xv_index]
            yv = inp[:, :, :, self.yv_index]
            xve = add_two_dims(xv)
            yve = add_two_dims(yv)
        if self.learn_density:
            densities = inp[:, :, :, self.densities_index]
            de  = add_two_dims(densities)
        else:
            print "FIXED DENSITY FOR MIXTURE GAUSSIANS!"
            de = 1.0

        xi = tf.linspace(0.0, 1.0, sizeX)
        xi = tf.reshape(xi, [1, 1, 1, -1, 1])
        xi = tf.tile(xi, [1, 1, k, 1, sizeY])
        # -> xi.shape==(1, k, sizeX, sizeY), xi[0][0] has #sizeX different rows, each col has #sizeY identical numbers in it.
        yi = tf.linspace(0.0, 1.0, sizeY)
        yi = tf.reshape(yi, [1, 1, 1, 1, -1])
        yi = tf.tile(yi, [1, 1, k, sizeX, 1])
        

        if self.learn_variance:
            error = (xi - xse) ** 2 / (xve * self.variance) + (yi - yse) ** 2 / (yve * self.variance)
        else:
            error = (xi - xse) ** 2 / self.variance + (yi - yse) ** 2 / self.variance
        error /= 2
        error = tf.minimum(error, 1)
        error = tf.maximum(error, -1)

        # avgpooling is better for reconstruction (if negative ds are allowed),
        # val_loss: 0.0068, but way-way worse for interpolation, it looks like a smoke monster.
        # Note that fixed variance maxpooling will never generalize beyond MNIST.
        if self.maxpooling:
            out = K.max(de * K.exp(-error), axis=2)
        else:
            out = K.sum((2 * de - 1) * K.exp(-error), axis=2)
        out = tf.transpose(out, [0, 2, 3, 1])
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.sizeX, self.sizeY, input_shape[1])


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

    celeba = np.load("celeba6472.npy").astype(np.float32)
    celeba = celeba[:, 4:-4, :]

    X_train, X_test = celeba[:50000], celeba[50000:55000]
    X_train = X_train.reshape(len(X_train), -1)
    X_test  = X_test .reshape(len(X_test ), -1)

    return image_size, (X_train, np.zeros(len(X_train), dtype=np.int32)), (X_test, np.zeros(len(X_test), dtype=np.int32))


def vis_landmarks(imgs_orig, landmarks, prefix):
    assert len(imgs_orig)==len(landmarks)
    assert landmarks.shape[1:] == (5, 2)
    imgs = np.copy(imgs_orig)
    for i in range(len(imgs)):
        face = imgs[i]
        marks = landmarks[i]
        for x, y in marks:
            x, y = int(x), int(y)
            face[y-1:y+2, x-1:x+2] = 1.0
    import vis
    vis.plotImages(np.expand_dims(imgs, 3), 10, 10, prefix)


def test_landmarks():
    shape = (72, 64)
    celeba_train, celeba_test = data.load_celeba(shape=shape, color=False)
    landmarks_train, landmarks_test = data.load_landmarks(shape=shape)

    landmarks_train[:, :, 0] /= 64
    landmarks_train[:, :, 1] /= 72
    landmarks_test [:, :, 0] /= 64
    landmarks_test [:, :, 1] /= 72

    X_train_flat = celeba_train.reshape((len(celeba_train), -1))
    y_train_flat = landmarks_train.reshape((len(landmarks_train), -1))
    X_test_flat = celeba_test.reshape((len(celeba_test), -1))
    y_test_flat = landmarks_test.reshape((len(landmarks_test), -1))

    batch_size = 128
    epochs = 10
    intermediate_layer_size = 200
    nonlinearity = "relu"
    inputs = Input(shape=(shape[0]*shape[1],))
    net = inputs
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(2*5, activation="sigmoid")(net)

    model = Model(input=inputs, output=net)

    model.summary()

    model.compile(loss='mse', optimizer=Adam())

    history = model.fit(X_train_flat, y_train_flat,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test_flat, y_test_flat))

    y_test_predicted = model.predict(X_test_flat).reshape(-1, 5, 2)
    y_test_predicted[:, :, 0] *= 64
    y_test_predicted[:, :, 1] *= 72
    print y_test_predicted[10:20]
    vis_landmarks(celeba_test, y_test_predicted, "predicted")
    vis_landmarks(celeba_test, y_test_predicted[::-1], "predicted_bad")


def test_learn():
    data_source = "celebabw"

    assert data_source in ("mnist", "celebabw")
    if data_source == "mnist":
        image_size, (X_train, y_train), (X_test, y_test) = load_mnist()
    elif data_source == "celebabw":
        image_size, (X_train, y_train), (X_test, y_test) = load_celebabw()

    train_size = None # 10000
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]

    nb_features = image_size * image_size

    batch_size = 32
    epochs = 10
    k = 600
    nonlinearity = 'relu'
    intermediate_layer_size = 1000

    if data_source == "mnist":
        learn_variance = False
        variance = 0.0005
	learn_density = False
    elif data_source == "celebabw":
        learn_variance = True
        variance = 1.0/200 # Interpreted as maximum allowed SD 7% of image size.
	learn_density = True
    maxpooling = False

    mixture_layer = MixtureLayer(image_size, image_size, learn_density=learn_density,
	    learn_variance=learn_variance, variance=variance, maxpooling=maxpooling)

    inputs = Input(shape=(nb_features,))
    net = inputs
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size / 2, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size / 2, activation=nonlinearity)(net)

    gauss_param_count = get_param_count(learn_variance, learn_density)

    net = Dense(k * gauss_param_count, activation='sigmoid')(net)
    gaussians = Reshape((1, k, gauss_param_count))(net)
    # input_shape = (batch, channels, dots, gauss_param_count)
    net = mixture_layer(gaussians)
    net = Reshape((nb_features,))(net)
    model = Model(input=inputs, output=net)

    model.summary()

    model.compile(loss='mse', optimizer=Adam())

    history = model.fit(X_train, X_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, X_test))

    saveModel(model, "model")

    # n = 400
    # displaySet(X_train[:n].reshape(n, image_size, image_size), n, model, "ae-train", flatten_input=True)
    # displaySet(X_test [:n].reshape(n, image_size, image_size), n, model, "ae-test",  flatten_input=True)

    encoder = Model(input=inputs, output=gaussians)
    encoder.compile(loss='mse', optimizer=SGD())

    input_gaussians = Input(shape=(1, k, gauss_param_count))
    output_image_size = image_size * 2 # We can increase the resolution
    mixture_layer_2 = MixtureLayer(output_image_size, output_image_size, learn_density=learn_density,
	    learn_variance=learn_variance, variance=variance, maxpooling=maxpooling)
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
# test_landmarks()

