import numpy as np
import os


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
