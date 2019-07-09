import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
import vgg19

weight_init_before = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

        return x

def deconv(x, channels, kernel=3, stride=2, use_bias=True, scope='deconv_0') :
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                       kernel_size=kernel, kernel_initializer=weight_init,
                                       kernel_regularizer=weight_regularizer,
                                       strides=stride, use_bias=use_bias, padding='SAME')

        return x

def linear(x, units, use_bias=True, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

        return x + x_init


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def group_norm(x, scope='group_norm'):
    #https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/layers/python/layers/normalization.py
    return tf_contrib.layers.group_norm(x)

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(type, real, fake):
    real_loss = 0
    fake_loss = 0

    if type == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if type == 'gan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    loss = real_loss + fake_loss

    return loss


def generator_loss(type, fake):
    fake_loss = 0

    if type == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if type == 'gan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    loss = fake_loss

    return loss


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss


def get_centre_mask(size):
    def normfun(x, mu, sigma):
        pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        return pdf

    # IMAGE_SIZE = 20
    inner_size = int(size / 2)
    pad_size = int((size - inner_size) / 2)
    line = np.arange(0, 1, 1 / inner_size)
    x = normfun(line, 0.5, 0.35)
    y = normfun(line, 0.5, 0.35)

    z = [xitem * y for xitem in x]
    z = np.maximum(z, 0)
    z = np.minimum(z, 1)
    z = np.pad(z, ((pad_size, size-inner_size-pad_size), (pad_size, size-inner_size-pad_size)), 'constant', constant_values=(0))

    return z


def get_centre_mask_tensor(size, batch_size):
    mask = get_centre_mask(size)
    mask = np.array(mask)
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
    mask = np.repeat(mask, 3, axis=2)

    mask_tensor = tf.constant(mask)
    mask_tensor = tf.tile(mask_tensor, multiples=[batch_size, 1, 1])  # fake.shape[0]
    mask_tensor = tf.reshape(mask_tensor, [batch_size, mask.shape[0], mask.shape[1], 3])
    mask_tensor = tf.cast(mask_tensor, dtype=tf.float32)
    return mask_tensor


def semantic_loss_with_attention(real, fake, batch_size):
    """"""
    vgg = vgg19.Vgg19('/home/benjamin/Workspace/ml/I19tModel/vgg19.npy')

    vgg.build(real)
    real_feature_map = vgg.conv3_3_no_activation

    mask_tensor = get_centre_mask_tensor(int(fake.shape[2]), batch_size)
    print("mask_tensor.shape = ", mask_tensor.shape)

    fake_masked = tf.multiply(mask_tensor, fake) + tf.multiply((1 - mask_tensor), real)

    vgg.build(fake_masked)
    fake_feature_map = vgg.conv3_3_no_activation

    loss = L1_loss(real_feature_map, fake_feature_map)

    return loss


def pix_loss_with_attention(real, fake, batch_size):
    mask_tensor = get_centre_mask_tensor(int(fake.shape[2]), batch_size)
    print("mask_tensor.shape = ", str(mask_tensor.get_shape().as_list()))
    fake_masked = tf.multiply(mask_tensor, fake) + tf.multiply((1 - mask_tensor), real)
    loss = L1_loss(real, fake_masked)
    return loss


def lsgan_loss_discriminator_counter_penalty(prob_penalty, batch_size):
    print("prob_penalty.shape=" + str(prob_penalty.get_shape().as_list()))
    mask_tensor = get_centre_mask_tensor(int(prob_penalty.shape[2]), batch_size)
    penalty_loss = tf.reduce_mean(tf.squared_difference(prob_penalty * (1-mask_tensor), 0))
    return penalty_loss
