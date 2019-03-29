import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class PrimitiveContext:

    def __init__(self, index, lookup_fn):
        self.index = index
        self.lookup_fn = lookup_fn

    def get_context(self):
        return self.lookup_fn(self.index)


class ApplyContext:

    def __init__(self, adj, base, apply_fn):
        self.adj = adj
        self.base = base
        self.apply_fn = apply_fn

    def get_context(self):
        return self.apply_fn(self.adj.get_context(), self.base.get_context())


class AndContext:

    def __init__(self, c1, c2, comp_fn):
        self.c1 = c1
        self.c2 = c2
        self.comp_fn = comp_fn

    def get_context(self):
        return self.comp_fn(self.c1.get_context(), self.c2.get_context())


class CompositionalParser:

    def __init__(self):
        self.shapes = [
            'square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle',
            'semicircle', 'ellipse', 'any'
        ]
        self.colors = [
            'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray', 'any'
        ]
        self.shape_embeddings = tf.get_variable(
            'shape_embeddings', [9, 8],
            initializer=tf.random_normal_initializer)
        self.color_embeddings = tf.get_variable(
            'color_embeddings', [8, 8, 8],
            initializer=tf.random_normal_initializer)
        self.dense1 = tf.layers.Dense(16)
        self.dense2 = tf.layers.Dense(8)
        self.apply_dense1 = tf.layers.Dense(16)
        self.apply_dense2 = tf.layers.Dense(8)

    def lookup_fn(self, index):
        return tf.cond(
            tf.greater(index, 8), lambda: tf.expand_dims(
                tf.nn.embedding_lookup(self.shape_embeddings, index), 0),
            lambda: tf.expand_dims(
                tf.nn.embedding_lookup(self.color_embeddings, index - 9), 0))

    def lookup_color_fn(self, index):
        return tf.expand_dims(tf.gather(self.color_embeddings, index - 9), 0)

    def lookup_shape_fn(self, index):
        return tf.expand_dims(tf.gather(self.shape_embeddings, index), 0)

    def comp_fn(self, c1, c2):
        c1 = self.dense1(c1)
        c2 = self.dense1(c2)
        return self.dense2(
            tf.nn.selu(tf.reduce_sum(tf.stack([c1, c2], axis=0), axis=0)))

    def apply_fn(self, adj, base):
        concat = tf.concat([adj, base], axis=-1)
        return self.apply_dense2(tf.nn.selu(self.apply_dense1(concat)))

    def matrix_apply_fn(self, adj, base):
        # x = tf.reshape(adj, [8, 8])
        print_op = tf.print({
            'adj': tf.math.count_nonzero(adj),
            'base': tf.math.count_nonzero(base)
        })
        with tf.control_dependencies([print_op]):
            return tf.matmul(adj[0], tf.transpose(base))[None, :]

    def parse_single_desc(self, desc):
        shape = PrimitiveContext(desc[0], self.lookup_shape_fn)
        color = PrimitiveContext(desc[1] + 1, self.lookup_color_fn)
        if desc[0] == -1:
            shape = PrimitiveContext(8, self.lookup_shape_fn)
        if desc[1] == -1:
            color = PrimitiveContext(16, self.lookup_color_fn)
        return ApplyContext(color, shape, self.matrix_apply_fn)

    def parse_multiple_descs(self, desc):
        return AndContext(
            self.parse_single_desc(desc[0, :]),
            self.parse_single_desc(desc[1, :]), self.comp_fn)

    def parse_apply(self, descs):
        return tf.map_fn(
            lambda x: tf.squeeze(self.parse_single_desc(x[0, :]).get_context()),
            descs,
            dtype=tf.float32)

    def parse_and(self, descs):
        return tf.map_fn(
            lambda x: tf.squeeze(self.parse_multiple_descs(x).get_context()),
            descs,
            dtype=tf.float32)

    def parse_descs(self, descs, apply=True):
        if apply:
            return self.parse_apply(descs)
        else:
            return self.parse_and(descs)
        # return tf.map_fn(
        #     lambda x: tf.squeeze(self.parse_multiple_descs(x)),
        #     descs,
        #     dtype=tf.float32)


class BasicParser:

    def __init__(self):
        self.elmo = hub.Module(
            "https://tfhub.dev/google/elmo/2", trainable=True)
        self.dense1 = tf.layers.Dense(8)

    def parse_descs(self, descs, apply=True):
        embeddings = self.elmo(
            descs, signature="default", as_dict=True)["default"]
        x = self.dense1(embeddings)
        return x
