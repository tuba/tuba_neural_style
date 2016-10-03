__author__ = 'alexander'

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
from libs import utils, gif, datasets, dataset_utils, vae, dft, vgg16, nb_utils

from libs import vgg16, inception, i2v

n_iterations = 100

# Experiment w/ different layers here.  You'll need to change this if you
# use another network!
content_layer = 'net/conv4_2/conv4_2:0'

# Experiment with different layers and layer subsets.  You'll need to change these
# if you use a different network!
style_layers = ['net/conv1_1/conv1_1:0',
                'net/conv2_1/conv2_1:0',
                'net/conv3_1/conv3_1:0',
                'net/conv4_1/conv4_1:0',
                'net/conv5_1/conv5_1:0']

device = '/gpu:0'
g = tf.Graph()
# Get the network again
net = vgg16.get_vgg_model()

# Preprocess images.
content_og = plt.imread('clinton.png')[..., :3]
style_og = plt.imread('arles.png')[..., :3]
content_img = net['preprocess'](content_og)[np.newaxis]
style_img = net['preprocess'](style_og)[np.newaxis]

content_features = None
style_features = []

with tf.Session(graph=g) as sess, g.device(device):

    # Now load the graph_def, which defines operations and their values into `g`
    tf.import_graph_def(net['graph_def'], name='net')

    # Grab the tensor defining the input to the network
    x = g.get_tensor_by_name("net/images:0")

    content_features = g.get_tensor_by_name(content_layer).eval(
        session=sess,
        feed_dict={x: content_img,
                   'net/dropout_1/random_uniform:0': [[1.0]],
                   'net/dropout/random_uniform:0': [[1.0]]})

    for style_i in style_layers:
        style_activation_i = g.get_tensor_by_name(style_i).eval(
            session=sess,
            feed_dict={x: style_img,
                       'net/dropout_1/random_uniform:0': [[1.0]],
                       'net/dropout/random_uniform:0': [[1.0]]})
        s_i = np.reshape(style_activation_i, [-1, style_activation_i.shape[-1]])
        gram_matrix = np.matmul(s_i.T, s_i) / s_i.size
        style_features.append(gram_matrix.astype(np.float32))

tf.reset_default_graph()
g = tf.Graph()
net = vgg16.get_vgg_model()

# Load up a session which we'll use to import the graph into.
with tf.Session(graph=g) as sess, g.device(device):
    # We can set the `net_input` to our content image
    # or perhaps another image
    # or an image of noise
    # net_input = tf.Variable(content_img / 255.0)
    net_input = tf.get_variable(
        name='input',
        shape=content_img.shape,
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(mean=np.mean(content_img), stddev=np.std(content_img)))

    # Now we load the network again, but this time replacing our placeholder
    # with the trainable tf.Variable
    tf.import_graph_def(net['graph_def'], name='net', input_map={'images:0': net_input})

    content_loss = tf.nn.l2_loss((g.get_tensor_by_name(content_layer) - content_features) / content_features.size)
    style_loss = np.float32(0.0)

    for style_layer_i, style_gram_i in zip(style_layers, style_features):
        layer_i = g.get_tensor_by_name(style_layer_i)
        layer_shape = layer_i.get_shape().as_list()
        layer_size = layer_shape[1] * layer_shape[2] * layer_shape[3]
        layer_flat = tf.reshape(layer_i, [-1, layer_shape[3]])
        gram_matrix = tf.matmul(tf.transpose(layer_flat), layer_flat) / layer_size
        style_loss = tf.add(style_loss, tf.nn.l2_loss((gram_matrix - style_gram_i) / np.float32(style_gram_i.size)))

    h, w = net_input.get_shape().as_list()[1], net_input.get_shape().as_list()[1]
    dx = tf.square(net_input[:, :h - 1, :w - 1, :] - net_input[:, :h - 1, 1:, :])
    dy = tf.square(net_input[:, :h - 1, :w - 1, :] - net_input[:, 1:, :w - 1, :])
    tv_loss = tf.reduce_sum(tf.pow(dx + dy, 1.25))


# TRAIN
with tf.Session(graph=g) as sess, g.device(device):
    # Experiment w/ the weighting of these!  They produce WILDLY different
    # results.
    loss = 5.0 * content_loss + 1.0 * style_loss + 0.001 * tv_loss
    optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)

imgs = []
with tf.Session(graph=g) as sess, g.device(device):
    sess.run(tf.initialize_all_variables())

    # map input to noise
    og_img = net_input.eval()

    for it_i in range(n_iterations):
        _, this_loss, synth = sess.run([optimizer, loss, net_input], feed_dict={
            'net/dropout_1/random_uniform:0': np.ones(
                g.get_tensor_by_name('net/dropout_1/random_uniform:0').get_shape().as_list()),
            'net/dropout/random_uniform:0': np.ones(
                g.get_tensor_by_name('net/dropout/random_uniform:0').get_shape().as_list())
        })
        print("%d: %f, (%f - %f)" %
              (it_i, this_loss, np.min(synth), np.max(synth)))
        if it_i % 5 == 0:
            m = vgg16.deprocess(synth[0])
            imgs.append(m)

    gif.build_gif(imgs, saveto='stylenet.gif')
