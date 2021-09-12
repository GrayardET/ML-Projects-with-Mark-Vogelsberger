# %%
import pickle
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
num_classes = 4
image_size = 112
in_channel = 3
n_epoches = 300
IMG_PATH = '/home/tai/XDF_Project/MIT-Reserach/Generative_Adversial_Nets/local_images/'
ARRAY_GIRL_PATH = r'/home/tai/XDF_Project/MIT-Reserach/Generative_Adversial_Nets/array_girl'
LABEL_GIRL_PATH = r'/home/tai/XDF_Project/MIT-Reserach/Generative_Adversial_Nets/label_girl'
color_set = [1, 2, 5, 9]
latent_vector_fixed = tf.random.normal(shape=(num_classes*20, latent_dim))
labels_fixed = []
for i in range(num_classes):
    labels_fixed.append(tf.ones((20, 1), dtype=tf.int32) * i)
labels_fixed = tf.concat(labels_fixed, axis=0)


# %%
with open(ARRAY_GIRL_PATH, 'rb') as file:
    girl_arrays = pickle.load(file)
file.close()
with open(LABEL_GIRL_PATH, 'rb') as file:
    girl_labels = pickle.load(file)
file.close()

idx = np.where((girl_labels == 1) | (girl_labels == 2) | (girl_labels == 5) | (girl_labels == 9))
girls = girl_arrays[idx].astype("float32") / 255.0 * 2 - 1
labels = girl_labels[idx]
for i in range(len(labels)):
    if labels[i] == 1:
        labels[i] = 0
    if labels[i] == 2:
        labels[i] = 1
    if labels[i] == 5:
        labels[i] = 2
    if labels[i] == 9:
        labels[i] = 3
dataset = tf.data.Dataset.from_tensor_slices((girls, labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

print(f"Shape of training images: {girls.shape}")
print(f"Shape of training labels: {labels.shape}")

# %%
# Generate the time stamp
def time_stamp():
    return datetime.datetime.now().isoformat()

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
    for i in range(num_classes):
        labels.append(tf.ones((10, 1), dtype=tf.int32) * i)
    labels = tf.concat(labels, axis=0)
    noises = tf.random.normal((num_classes*10, latent_dim))

    img = model([noises, labels]).numpy().squeeze()

    rows = []
    for i in range(num_classes):
        row = np.concatenate(img[i*10: i*10+10], axis=1)
        rows.append(row)
    img = np.concatenate(rows, axis=0)
    img = (img + 1) / 2

    figure, ax = plt.subplots(figsize=(50, 50))

    ax.imshow(img)
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
def define_discriminator(in_shape=(112,112,3), n_classes=4):
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
	fe = Conv2D(128, (5,5), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1)(fe)
	# define model
	model = Model([in_image, in_label], out_layer, name='discriminator')
	# compile model
	# opt = Adam(lr=0.0002, beta_1=0.5)
	# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model

# define the standalone generator model
def define_generator(latent_dim=128, n_classes=4):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 100)(in_label)
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
    # upsample to 56x56
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 112x112
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(3, (7,7), activation='tanh', padding='same')(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)

    return model

# %%
class WGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, discriminator_extra_steps=3, gp_weight=10.0):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

        return

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

        return

    def gradient_penalty(self, batch_size, real_images, real_labels, fake_images, fake_labels):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, real_labels], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated, real_labels])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    def train_step(self, data):
        real_images, real_labels = data

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            # Get the random class labels
            random_labels = tf.random.uniform(
                shape=(batch_size, 1),
                minval=0,
                maxval=num_classes,
                dtype=tf.int32
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, random_labels], training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator([fake_images, random_labels], training=True)
                # Get the logits for the real images
                real_logits = self.discriminator([real_images, real_labels], training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, real_labels, fake_images, random_labels)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # Get the random class labels
        random_labels = tf.random.uniform(
            shape=(batch_size, 1),
            minval=0,
            maxval=num_classes,
            dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, random_labels], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([generated_images, random_labels], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}
    
class GANMonitor(keras.callbacks.Callback):
    def __init__(self):
        return

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator([latent_vector_fixed, labels_fixed])
        generated_images = (generated_images * 127.5) + 127.5
        generated_images = generated_images.numpy()

        rows = []
        for i in range(num_classes * 2):
            row = []
            for j in range(10):
                row.append(generated_images[i*10+j])
            row = np.concatenate(row, axis=1)
            rows.append(row)
        img = np.concatenate(rows, axis=0)

        img = keras.preprocessing.image.array_to_img(img)
        img.save(
            IMG_PATH+time_stamp()+":generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch)
            )
        
        return
# %%
d_model = define_discriminator(
    in_shape=(112, 112, 3),
    n_classes=4
    )
d_model.summary()
g_model = define_generator(
    latent_dim=latent_dim,
    n_classes = 4
    )
g_model.summary()
# create the discriminator
# d_model = define_discriminator()
# # create the generator
# g_model = define_generator(latent_dim)
# # create the gan
# gan_model = define_gan(g_model, d_model)
# # load image data
# # train model
# train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=300, display=True)
# Instantiate the optimizer for both networks
# (learning_rate=0.0002, beta_1=0.5 are recommended)
generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

# Instantiate the customer `GANMonitor` Keras callback.
cbk = GANMonitor()

# Instantiate the WGAN model.
wgan = WGAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=latent_dim,
    discriminator_extra_steps=3,
)

# Compile the WGAN model.
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

# Start training the model.
wgan.fit(dataset, batch_size=batch_size, epochs=n_epoches, callbacks=[cbk])

# %%
for data in dataset:
    x, y = data
    break
f = tf.random.normal((64, 112, 112, 3))
r = tf.random.normal((64, 112, 112, 3))

# %%
alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
diff = f - r
interpolated = r + alpha * diff
# %%
interpolated.shape
# %%
