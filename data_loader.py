import numpy as np
import os
import tensorflow as tf


def load_data(folder):
    inputs = np.load(os.path.join(folder, 'inputs.npy'))
    examples = np.load(os.path.join(folder, 'examples.npy'))
    examples = examples.reshape(-1, 64, 64, 3)
    return np.concatenate([inputs, examples], axis=0)


def load_data_separate(folder):
    inputs = np.load(os.path.join(folder, 'inputs.npy'))
    examples = np.load(os.path.join(folder, 'examples.npy'))
    examples = examples.reshape(-1, 64, 64, 3)
    return inputs, examples


def load_all_data(folder, split_descs=False):
    descriptions = np.load(
        os.path.join(folder,
                     'hints' + ('_split' if split_descs else '') + '.npy'))
    if split_descs:
        lengths = np.array([[len(elem[0].split(' ')),
                             len(elem[1].split(' '))] for elem in descriptions])
    else:
        lengths = np.array([len(desc.split(' ')) for desc in descriptions])
    examples = np.load(os.path.join(folder, 'examples_mnet.npy'))
    inputs = np.float32(np.load(os.path.join(folder, 'inputs.npy')))
    labels = np.int32(np.load(os.path.join(folder, 'labels.npy')))
    return descriptions, lengths, examples, inputs, labels


def fon1(arr):
    return -1 if len(arr) is 0 else arr[0]


def parse(f):
    shapes = [
        'square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle',
        'semicircle', 'ellipse'
    ]
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray']
    descs = np.load(f)
    parsed = np.full((descs.shape[0], 2, 2), -1)
    for jx, desc in enumerate(descs):
        shape1 = [ix for ix, shape in enumerate(shapes) if shape in desc[0]]
        shape2 = [ix for ix, shape in enumerate(shapes) if shape in desc[1]]
        color1 = [ix + 8 for ix, color in enumerate(colors) if color in desc[0]]
        color2 = [ix + 8 for ix, color in enumerate(colors) if color in desc[1]]
        parsed[jx, :, :] = [[fon1(shape1), fon1(color1)],
                            [fon1(shape2), fon1(color2)]]
    return parsed


def load_dil(folder):
    inputs = np.float32(np.load(os.path.join(folder, 'inputs.npy')))
    labels = np.int32(np.load(os.path.join(folder, 'labels.npy')))
    descs = parse(os.path.join(folder, 'hints_split.npy'))
    return inputs, labels, descs


def parse_example(example_proto):
    feature_description = {
        'image': tf.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.FixedLenFeature([], tf.int64, default_value=0),
        'concept': tf.FixedLenFeature([], tf.string, default_value=''),
        # 'full_caption': tf.FixedLenFeature([], tf.string, default_value=''),
    }
    features = tf.parse_single_example(example_proto, feature_description)
    return tf.image.decode_png(features['image'],
                               3), features['label'], features['concept']


def create_dataset(filename):

    raw_dataset = tf.data.TFRecordDataset(filename)
    raw_dataset = raw_dataset.map(parse_example)

    return raw_dataset


def create_input_parser(comp=False):
    if comp:
        vocab = [
            'square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle',
            'semicircle', 'ellipse', 'red', 'green', 'blue', 'yellow',
            'magenta', 'cyan', 'gray'
        ]
        table = tf.contrib.lookup.index_table_from_tensor(mapping=vocab)

    def parse(image, label, desc_string):
        desc_string = tf.string_strip(desc_string)
        if comp:
            desc_string = tf.string_split([desc_string], ',').values
            desc_string = tf.string_split(desc_string)
            desc_string = tf.sparse_tensor_to_dense(
                table.lookup(desc_string), default_value=-1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image.set_shape([64, 64, 3])
        # label = tf.cast(label, tf.int32)
        return image, label, desc_string

    return parse
