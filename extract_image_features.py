import tensorflow as tf
import tensorflow_hub as hub
from data_loader import load_data_separate
import numpy as np
import os
IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96
NUM_FEATURES = 1280
def extract_features():
    # train_inputs, train_examples = load_data_separate('complearn/train/')
    # val_inputs, val_examples = load_data_separate('complearn/val/')
    # test_inputs, test_examples = load_data_separate('complearn/test/')
    # data = np.concatenate([train_data, val_data, test_data], axis=0)
    with tf.Graph().as_default():
        module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/2")
        height, width = hub.get_expected_image_size(module)
        print('model downloaded')
        images = tf.placeholder(tf.float32, [None, 64, 64, 3])
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.batch(128)
        dataset = dataset.map(lambda image: tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH]))
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
    
        # images =   # A batch of images with shape [batch_size, height, width, 3].
        features = module(next_element)  # Features with shape [batch_size, num_features].

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            for folder in ['complearn/train/',
                           'complearn/val/',
                           'complearn/test/']:
                inputs, examples = load_data_separate(folder)
                sess.run(iterator.initializer, feed_dict={images: inputs})
                feats = []
                while True:
                    try:
                        image_features = sess.run(features)
                        feats.append(image_features)
                    except tf.errors.OutOfRangeError:
                        break
                feats = np.concatenate(feats, axis=0)
                np.save(os.path.join(folder, 'input_mnet.npy'), feats)
                
                feats = []
                sess.run(iterator.initializer, feed_dict={images: examples})
                while True:
                    try:
                        image_features = sess.run(features)
                        feats.append(image_features)
                    except tf.errors.OutOfRangeError:
                        break
                feats = np.concatenate(feats, axis=0)
                feats = feats.reshape(-1, 4, NUM_FEATURES)
                np.save(os.path.join(folder, 'examples_mnet.npy'), feats)

                print('extracted ' + folder)

if __name__ == '__main__':
    extract_features()                