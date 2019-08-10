import tensorflow as tf


class FCGenerator:
    def __init__(self, img_size, channels):
        """
        Network which takes a batch of random vectors and creates images out of them with.

        :param img_size: width and height of the image
        :param channels: number of channels
        """
        self.img_size = img_size
        self.channels = channels

    def __call__(self, z):
        """
        Method which performs the computation.

        :param z: tensor of the shape [batch_size, z_size] representing batch_size random vectors from the
        prior distribution
        :return: image of the shape [batch_size, img_size, img_size, channels]
        """
        with tf.variable_scope("Generator"):
            z = tf.layers.dense(z, 512, activation=tf.nn.relu)
            z = tf.layers.dense(z, 512, activation=tf.nn.relu)
            z = tf.layers.dense(z, self.img_size * self.img_size * self.channels, activation=tf.nn.sigmoid)
            image = tf.reshape(z, [-1, self.img_size, self.img_size, self.channels])
            return image


class ConvGenerator:
    def __init__(self, img_size, channels):
        self.img_size = img_size
        self.channels = channels

    def __call__(self, z):
        with tf.variable_scope("Generator"):
            act = tf.nn.relu
            res_met = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            pad2 = [[0, 0], [2, 2], [2, 2], [0, 0]]

            kwargs = {"strides": (1, 1), "padding": "valid"}

            z = tf.layers.dense(z, 32768, activation=act)
            z = tf.reshape(z, [-1, 4, 4, 2048])

            z = tf.pad(z, pad2, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=1024, kernel_size=(5, 5), **kwargs, activation=act)
            z = tf.image.resize_images(z, (16, 16), method=res_met)
            #
            z = tf.pad(z, pad2, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=512, kernel_size=(5, 5), **kwargs, activation=act)
            z = tf.image.resize_images(z, (32, 32), method=res_met)

            z = tf.pad(z, pad2, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=256, kernel_size=(5, 5), **kwargs, activation=act)
            z = tf.image.resize_images(z, (self.img_size, self.img_size), method=res_met)

            z = tf.pad(z, pad2, mode="SYMMETRIC")
            z = tf.layers.conv2d(z, filters=3, activation=tf.nn.sigmoid, kernel_size=(5, 5), **kwargs)
            return z


class DCGANGenerator:
    def __init__(self, img_size, channels):
        self.channels = channels

    def __call__(self, z, prev_x):
        """

        :param z:
        :return: returns tensor of shape [batch_size, 64, 64, channels]
        """
        with tf.variable_scope("Generator"):
            act = tf.nn.leaky_relu

            h0 = tf.layers.conv2d(prev_x, filters=16, kernel_size = (1, 128), strides = (2, 2), padding= valid, activation=act)
            h1 = tf.layers.conv2d(h0, filters=16, kernel_size = (2, 2), strides = (2, 2), padding= valid, activation=act)
            h2 = tf.layers.conv2d(h1, filters=16, kernel_size = (2, 2), strides = (2, 2), padding= valid, activation=act)
            h3 = tf.layers.conv2d(h2, filters=16, kernel_size = (2, 2), strides = (2, 2), padding= valid, activation=act)
            h4 = tf.layers.dense(h3, 1)

            z = tf.layers.dense(z, 1024, activation=act)
            z = tf.layers.dense(z, 256, activation=act)
            z = tf.reshape(z, [-1, 2, 1, 128])
            z = tf.concat(z, h4)

            kwargs = {"kernel_size": (2, 1), "strides": (2, 2), "padding": "valid"}

            z = tf.layers.conv2d_transpose(z, output_shape =(4,1) , activation=act, **kwargs)
            z = tf.concat(z, h2)
            z = tf.layers.conv2d_transpose(z, output_shape =(8,1), activation=act, **kwargs)
            z = tf.concat(z, h1)
            z = tf.layers.conv2d_transpose(z, output_shape =(16,1), activation=act, **kwargs)
            z = tf.concat(z, h0)
            z = tf.layers.conv2d_transpose(z, output_shape =(16,128), activation=tf.nn.sigmoid, kernel_size = (1, 128), strides = (1, 2))
            return z
