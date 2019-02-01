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
    descriptions = np.load(os.path.join(folder, 'hints' + ('_split' if split_descs else '') + '.npy'))
    if split_descs:
        lengths = np.array([[len(elem[0].split(' ')), len(elem[1].split(' '))] for elem in descriptions])
    else:
        lengths = np.array([len(desc.split(' ')) for desc in descriptions])
    examples = np.load(os.path.join(folder, 'examples_mnet.npy'))
    inputs = np.load(os.path.join(folder, 'input_mnet.npy'))
    labels = np.load(os.path.join(folder, 'hints.npy'))
    return descriptions, lengths, examples, inputs, labels