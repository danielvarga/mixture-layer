from keras.datasets import mnist
import os
import os.path
from PIL import Image
import numpy as np

def load(dataset, shape=None, color=False):
    if dataset == "mnist":
        assert shape is None
        assert color is False
        (x_train, x_test) = load_mnist()
    elif dataset == "celeba":
        # What's the pythonic way of doing this?
        (x_train, x_test) = load_celeba(shape=shape, color=color)  if shape is not None else load_celeba(color=color)
    else:
        raise Exception("Invalid dataset: ", dataset)

    # if the feature dimension is missing, add it (!!! only works with Tensorflow !!!)
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, 3)
        x_test = np.expand_dims(x_test, 3)

#    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return (x_train, x_test)

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    return (x_train, x_test)

def load_celeba(shape=(72, 64), color=False):
    if shape==(72, 60):
        directory = "/home/daniel/autoencoding_beyond_pixels/datasets/celeba/img_align_celeba-60x72"
        if color:
            cacheFile = "/home/zombori/datasets/celeba_color.npy"
        else:
            cacheFile = "/home/csadrian/datasets/celeba.npy"
    elif shape==(72, 64):
        directory = "/home/daniel/autoencoding_beyond_pixels/datasets/celeba/img_align_celeba-64x72"
        if color:
            cacheFile = "/home/zombori/datasets/celeba6472_color.npy"
        else:
            cacheFile = "celeba6472.npy"
    else:
        assert False, "We don't have a celeba dataset with this size. Maybe you forgot about height x width order?"

    trainSize = 50000
    testSize = 5000
    if os.path.isfile(cacheFile):
        input = np.load(cacheFile)
    else:
        imgs = []
        height = None
        width = None
        for f in sorted(os.listdir(directory)):
            if len(imgs) % 5000 == 0:
                print len(imgs)
            if len(imgs) >= trainSize+testSize:
                break
            if f.endswith(".jpg") or f.endswith(".png"):
                if color:
                    img = Image.open(os.path.join(directory, f))
                else:
                    img = Image.open(os.path.join(directory, f)).convert("L")                
                arr = np.array(img)
                if height is None:
                    height, width = arr.shape[:2]
                else:
                    assert (height, width) == arr.shape[:2], "Bad size %s %s" % (f, str(arr.shape))
                imgs.append(arr)
        input = np.array(imgs)
        input = input[:trainSize + testSize] / 255.0 # the whole dataset does not fit into memory as a float
        np.save(cacheFile,input)

    x_train = input[:trainSize]
    x_test = input[trainSize:trainSize+testSize]
    np.random.seed(10) # TODO Not the right place to do this.
    x_train = np.random.permutation(x_train)
    x_test = np.random.permutation(x_test)
    return (x_train, x_test)


def load_landmarks(shape=(72, 64)):
    filename = "/home/daniel/experiments/mixture-layer/mixture-layer/list_landmarks_align_celeba.txt"
    vs = []
    for i, l in enumerate(file(filename)):
        if i<2:
            continue
        # lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
        a = l.strip().split()
        assert a[0].endswith(".jpg")
        v = np.array(a[1:]).reshape((5, 2))
        vs.append(v)
        if len(vs)>=55000:
            break
    vs = np.array(vs).astype(np.float32)
    vs[:, :, 0] /= 178
    vs[:, :, 1] /= 218
    vs[:, :, 0] *= 60
    vs[:, :, 0] +=  2
    vs[:, :, 1] *= 72
    return vs[:50000], vs[50000:55000]
