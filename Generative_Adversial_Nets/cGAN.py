# %%
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
from tqdm import tqdm
from functools import wraps

gpus = tf.config.list_physical_devices('GPU')

# I have only one GPU whic is needed to support my screen
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

batch_size = 64
latent_dim = 128
num_classes = 10
image_size = 28
IMG_PATH = '/home/tai/XDF_Project/MIT-Reserach/Generative_Adversial_Nets/local_images/'

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# Scale the pixel values to [-1, 1] range, add a channel dimension
all_digits = all_digits.astype("float32") / 255.0 * 2 - 1
all_digits = np.reshape(all_digits, (-1, image_size, image_size, 1))

# Create tf.data.Dataset.
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")

# %%
# the decorator to close figure after save
def close_figure(func):
    @wraps(func)
    def plot_with_close(*args, **kwargs):
        back = func(*args, **kwargs)
        plt.clf()
        plt.cla()
        plt.close()
        
        return back
    return plot_with_close

# plot 9 figures within one class
@close_figure
def plot_class(model, class_num):
    label = class_num
    label = tf.convert_to_tensor(label)
    label = tf.reshape(label, (1, 1))

    _, ax = plt.subplots(3, 3, figsize=(20, 20))
    for i in range(3):
        for j in range(3):
            noise = tf.random.normal(shape=(1, latent_dim))
            img = model([noise, label]).numpy().squeeze()

            ax[i, j].imshow(img, cmap='gray_r')
            ax[i, j].axis('off')

    plt.show()
    return

# plot images from all classes, and save it if the title is given
@close_figure
def plot_all_class(model, title=None):
    if title is None:
        plt.ion()
    else: 
        plt.ioff()

    labels = []
    for i in range(10):
        labels.append(tf.ones((10, 1)) * i)

    labels = tf.concat(labels, axis=0)
    noises = tf.random.normal((num_classes*10, latent_dim))

    img = model([noises, labels]).numpy().squeeze()

    rows = []
    for i in range(10):
        row = np.concatenate(img[i*10: i*10+10], axis=1)
        rows.append(row)
    img = np.concatenate(rows, axis=0)

    figure, ax = plt.subplots(figsize=(50, 50))

    ax.imshow(img, cmap='gray_r')
    ax.axis('off')

    if title is None:
        plt.show()
    else:
        figure.savefig(
            fname=title + '.png',
        )

    return

# plot the training process
@close_figure
def plot_training(g_hist, d_hist, title=None):
    if title is None:
        plt.ion()
    else:
        plt.ioff()

    figure, ax = plt.subplots()
    ax.plot(g_hist, label='Generator')
    ax.plot(d_hist, label='Discriminator')
    ax.set(xlabel='Epoch')
    ax.set(ylabel='Loss')

    if title is None:
        plt.show()
    else:
        figure.suptitle('Training Losses')
        figure.savefig(title + '.png')

    return
    
# %%
# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1), n_classes=10):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=10):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 7 * 7
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((7, 7, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)

	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)

	return model

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, display=False):
	# manually enumerate epochs
    d_loss_hist = []
    g_loss_hist = []
    for i in tqdm(range(n_epochs)):
        j = 1
        if display and (j - 1) % 2 == 0:
            title = IMG_PATH + datetime.datetime.now().isoformat() + "epoch:%d" % (i + 1)
            plot_all_class(g_model, title)
        # enumerate batches over the training set
        for batch in dataset:
            # load the data, find the current batch size
            (X_real, labels_real) = batch
            batch_size = labels_real.shape[0]
            # one vector for real images
            y_real = tf.ones((batch_size, 1))

            # update discriminator model weights on real images
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)

            # generate random noise
            noise = tf.random.normal((batch_size, latent_dim))
            # gererate random class labels
            labels_fake = tf.random.uniform((batch_size, 1), minval=0, maxval=num_classes-1, dtype=tf.int32)
            # generate fake images
            X_fake = g_model([noise, labels_fake])
            # zero vector for fake images
            y_fake = tf.zeros((batch_size, 1))

            # update discriminator model weights on fake images
            d_loss2, _ = d_model.train_on_batch([X_fake, labels_fake], y_fake)

            # prepare points in latent space as input for the generator
            noise = tf.random.normal((batch_size, latent_dim))
            labels_fake = tf.random.uniform((batch_size, 1), minval=0, maxval=num_classes-1, dtype=tf.int32)
            # create inverted labels for the fake samples
            y_gan = tf.ones((batch_size, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([noise, labels_fake], y_gan)

            # count how many epochs
            j += 1

        # collect the losses of this epoch
        d_loss = (d_loss1 + d_loss2) / 2.0
        d_loss_hist.append(d_loss)
        g_loss_hist.append(g_loss)


    # save the generator model
    g_model.save('local_model/'+datetime.datetime.now().isoformat()+'cgan_generator.model')
    d_model.save('local_model/'+datetime.datetime.now().isoformat()+'cgan_discriminator.model')

    # plot and save the loss history
    plot_training(g_loss_hist, d_loss_hist, title=IMG_PATH + datetime.datetime.now().isoformat() + 'Loss')

    return

# %%
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
# train model
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=2, display=True)
