
# coding: utf-8

# In[6]:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
tfd = tf.contrib.distributions
from tensorflow.examples.tutorials.mnist import input_data
from data_loader import load_data

def plot_codes(ax, codes, labels):
  ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
  ax.set_aspect('equal')
  ax.set_xlim(codes.min() - .1, codes.max() + .1)
  ax.set_ylim(codes.min() - .1, codes.max() + .1)
  ax.tick_params(
      axis='both', which='both', left='off', bottom='off',
      labelleft='off', labelbottom='off')


def plot_samples(samples, epoch):
    fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(10, 1))
                    
    for index, sample in enumerate(samples):
        ax[index].imshow(sample)
        ax[index].axis('off')
    plt.savefig('vae-comp' + str(epoch) + '.png', dpi=300, transparent=True, bbox_inches='tight')

# In[18]:

def make_encoder(images, code_size):
    x = tf.layers.conv2d(images, 32, 3, padding='SAME')
    x = tf.layers.conv2d(x, 32, 3, padding='SAME')
    x = tf.layers.conv2d(x, 64, 3, padding='SAME')
    x = tf.layers.conv2d(x, 64, 3, padding='SAME')
    x = tf.reduce_mean(x, axis=[1,2])
    mean = tf.layers.dense(x, code_size)
    stddev = tf.layers.dense(x, code_size)
    return tfd.MultivariateNormalDiag(mean, stddev), mean
    # x = tf.layers.flatten(images)
    # x = tf.layers.dense(x, 200, tf.nn.relu)
    # x = tf.layers.dense(x, 200, tf.nn.relu)
    # loc = tf.layers.dense(x, code_size)
    # scale = tf.layers.dense(x, code_size, tf.nn.softplus)
    # return tfd.MultivariateNormalDiag(loc, scale), None

# In[17]:

def make_prior(code_size):
    loc = tf.zeros(code_size)
    scale = tf.ones(code_size)
    return tfd.MultivariateNormalDiag(loc, scale)


# In[24]:

def make_decoder(code, image_shape):
    num_units = image_shape[0] * image_shape[1] * image_shape[2]
    code = tf.layers.dense(code, num_units)
    code = tf.reshape(code, (-1, image_shape[0], image_shape[1], image_shape[2]))
    x = tf.layers.conv2d_transpose(code, 64, 3, padding='SAME')
    x = tf.layers.conv2d_transpose(x, 64, 3, padding='SAME')
    x = tf.layers.conv2d_transpose(x, 32, 3, padding='SAME')
    logits = tf.layers.conv2d_transpose(x, 3, 3, padding='SAME')
    return tfd.Independent(tfd.Bernoulli(logits), 3)
    # x = code
    # x = tf.layers.dense(x, 200, tf.nn.relu)
    # x = tf.layers.dense(x, 200, tf.nn.relu)
    # logit = tf.layers.dense(x, np.prod(image_shape))
    # logit = tf.reshape(logit, [-1] + image_shape)
    # return tfd.Independent(tfd.Bernoulli(logit), 2)


# In[11]:

make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)

# In[26]:

def vae(images):
    code_size = 32
    prior = make_prior(code_size)
    posterior, mean = make_encoder(images, code_size)
    code = posterior.sample()
    # likelihood = make_decoder(code, images.shape[1:]).log_prob(images)
    likelihood = make_decoder(code, [64, 64, 3]).log_prob(images)
    divergence = tfd.kl_divergence(posterior, prior)
    elbo = tf.reduce_mean(likelihood - divergence)
    samples = make_decoder(prior.sample(10), [64, 64, 3]).mean()

    return elbo, mean, code, samples
vae = tf.make_template('vae', vae)

# In[13]:

def run_training():
    EPOCHS = 200
    SAVE_EVERY = 10
    train_data = load_data('complearn/train/')
    val_data = load_data('complearn/val/')
    test_data = load_data('complearn/test/')
    # mnist = input_data.read_data_sets('MNIST_data/')

    print('data loaded')
    data = np.concatenate([train_data, val_data, test_data], axis=0)
    # data = mnist.train.images.reshape(-1, 28, 28, 1)
    # test = mnist.test.images.reshape(-1, 28, 28, 1)
    
    data_placeholder = tf.placeholder(tf.float32, [None, data.shape[1], data.shape[2], data.shape[3]])
    dataset = tf.data.Dataset.from_tensor_slices(data_placeholder)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(128)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    elbo, mean, code, samples = vae(next_element)
    optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


    print('Training started')
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(EPOCHS):
            sess.run(iterator.initializer, feed_dict={data_placeholder: data})
            while True:
                try:
                    sess.run([optimize])
                except tf.errors.OutOfRangeError:
                    break
            
            sess.run(iterator.initializer, feed_dict={data_placeholder: data})
            # plot_codes(ax[epoch, 0], test_codes, mnist.test.labels)
            if (epoch + 1) % SAVE_EVERY == 0:
                [loss, test_samples] = sess.run([elbo, samples])
                plot_samples(test_samples.reshape(-1, 64, 64, 3), epoch)
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model saved in path: %s" % save_path)
            else:
                [loss] = sess.run([elbo])
                
            print('Epoch %d elbo loss: %f' % (epoch, loss))

def extract_features():
    shape = [None, 64, 64, 3]
    data_placeholder = tf.placeholder(tf.float32, shape)
    dataset = tf.data.Dataset.from_tensor_slices(data_placeholder)
    dataset = dataset.batch(128)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    elbo, mean, code, samples = vae(next_element)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './models/model.ckpt')
        for data in ['train', 'val', 'test']:
            examples = np.load('complearn/' + data + '/examples.npy').reshape(-1, 64, 64, 3)
            inputs = np.load('complearn/' + data + '/inputs.npy')
            sess.run(iterator.initializer, feed_dict={data_placeholder: data})
            while True:
                try:
                    sess.run([mean])        
    return


# In[37]:

run_training()

