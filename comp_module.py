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
        return self.apply_fn(self.adj, self.base)


class AndContext:

    def __init__(self, c1, c2, comp_fn):
        self.c1 = c1
        self.c2 = c2
        self.comp_fn = comp_fn

    def get_context(self):
        return self.comp_fn(self.c1, self.c2)


class CompositionalParser:

    def __init__(self, hidden_dimension_size=16, use_glove=False):
        self.scope = tf.get_variable_scope()
        self.shapes = [
            'square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle',
            'semicircle', 'ellipse', 'any'
        ]
        self.colors = [
            'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray', 'any'
        ]
        self.use_glove = use_glove
        if use_glove:
            embedding = np.load('embeddings.npy')
            self.color_embeddings = tf.get_variable(
                'color_embeddings',
                initializer=embedding[8:, :].astype(np.float32),
                dtype=tf.float32)
            self.shape_embeddings = tf.get_variable(
                'shape_embeddings',
                initializer=embedding[:8, :].astype(np.float32),
                dtype=tf.float32)
            self.primitive_dense = tf.layers.Dense(8, name='primitive_dense')
        else:
            self.shape_embeddings = tf.get_variable(
                'shape_embeddings',
                # [8, 8],
                [9, 8],
                initializer=tf.random_normal_initializer)
            self.color_embeddings = tf.get_variable(
                'color_embeddings',
                # [7, 8],
                [8, 8],
                initializer=tf.random_normal_initializer)
        self.dense1 = tf.layers.Dense(hidden_dimension_size,
                                      name='dense1',
                                      use_bias=False)
        self.dense2 = tf.layers.Dense(8, name='dense2', use_bias=False)
        self.apply_dense1 = tf.layers.Dense(hidden_dimension_size,
                                            name='dense1',
                                            use_bias=False)
        self.apply_dense2 = tf.layers.Dense(8, name='dense2', use_bias=False)

    def lookup_fn(self, index):
        return tf.cond(
            tf.greater(index, 7), lambda: tf.expand_dims(
                tf.nn.embedding_lookup(self.shape_embeddings, index), 0),
            lambda: tf.expand_dims(
                tf.nn.embedding_lookup(self.color_embeddings, index - 8), 0))

    def lookup_color_fn(self, index):
        # print_op = tf.print({
        #     'index':
        #     index,
        #     'color_embeddings':
        #     self.color_embeddings,
        #     'curent_embedding':
        #     tf.expand_dims(tf.gather(self.color_embeddings, index - 9), 0)
        # })
        # with tf.control_dependencies([print_op]):
        # return tf.expand_dims(
        #     tf.nn.embedding_lookup(self.color_embeddings, index - 9), 0)
        embedding = tf.expand_dims(tf.gather(self.color_embeddings, index - 8),
                                   0)
        if self.use_glove:
            embedding = self.primitive_dense(embedding)
        return embedding

    def lookup_shape_fn(self, index):
        # print_op = tf.print({
        #     'index':
        #     index,
        #     'shape_embeddings':
        #     tf.count_nonzero(self.shape_embeddings),
        #     'curent_embedding':
        #     tf.count_nonzero(
        #         tf.expand_dims(tf.gather(self.shape_embeddings, index), 0))
        # })
        # with tf.control_dependencies([print_op]):
        # print('*************************' + str(index) + str(type(index)))
        # return tf.expand_dims(self.shape_embeddings[index, :], 0)
        # return tf.expand_dims(
        #     tf.nn.embedding_lookup(self.shape_embeddings, index), 0)
        embedding = tf.expand_dims(tf.gather(self.shape_embeddings, index), 0)
        if self.use_glove:
            embedding = self.primitive_dense(embedding)
        return embedding

    def comp_fn(self, c1, c2):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('comp'):

                c1 = self.dense1(c1)
                c2 = self.dense1(c2)
                return self.dense2(
                    tf.nn.selu(tf.reduce_sum(tf.stack([c1, c2], axis=0),
                                             axis=0)))

    def apply_fn(self, adj, base):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('apply'):
                concat = tf.concat([adj, base], axis=-1)
                return self.apply_dense2(tf.nn.selu(self.apply_dense1(concat)))

    def matrix_apply_fn(self, adj, base):
        return tf.matmul(adj[0], tf.transpose(base))[None, :]

    def parse_single_desc(self, desc):
        color_ind = desc[0]
        shape_ind = desc[1]
        shape = PrimitiveContext(shape_ind, self.lookup_shape_fn).get_context()
        color = PrimitiveContext(color_ind, self.lookup_color_fn).get_context()
        return ApplyContext(color, shape, self.apply_fn).get_context()

    def parse_multiple_descs(self, desc):
        return AndContext(self.parse_single_desc(desc[0, :]),
                          self.parse_single_desc(desc[1, :]),
                          self.comp_fn).get_context()

    def parse_apply(self, descs):
        return tf.map_fn(lambda x: tf.squeeze(self.parse_single_desc(x[0, :])),
                         descs,
                         dtype=tf.float32)

    def compute_and(self, i, desc, new_desc):
        and_embedding = AndContext(desc[2 * i:2 * i + 1],
                                   desc[2 * i + 1:2 * i + 2],
                                   self.comp_fn).get_context()
        new_desc = tf.concat([new_desc, and_embedding], 0)
        i = tf.add(i, 1)
        return i, desc, new_desc

    def parse_across(self, desc):
        num_iter = tf.floordiv(tf.shape(desc)[0], 2)
        i1 = tf.constant(0, dtype=tf.int32)
        new_descs = tf.constant([[]], shape=(0, 8))
        while_loop_op = tf.while_loop(
            cond=lambda i, desc, new_descs: tf.less(i, num_iter),
            body=self.compute_and,
            loop_vars=[i1, desc, new_descs],
            shape_invariants=[
                i1.get_shape(),
                desc.get_shape(),
                tf.TensorShape([None, 8])
            ])
        return while_loop_op[2]

    def parse_and(self, desc):
        desc = tf.reshape(desc, tf.constant([-1, 8]))
        return tf.while_loop(lambda desc: tf.greater(tf.shape(desc)[0], 1),
                             body=self.parse_across,
                             loop_vars=[desc],
                             shape_invariants=[tf.TensorShape([None, 8])],
                             parallel_iterations=1)

        # if tf.shape(desc)[0] > 1:
        #     return AndContext(self.parse_and(desc[:tf.shape(desc)[0] // 2, :]),
        #                       self.parse_and(desc[tf.shape(desc)[0] // 2:, :]),
        #                       self.comp_fn).get_context()
        # else:
        #     return self.parse_single_desc(desc[0, :])

    # def parse_and(self, desc):
    #     print(desc)
    #     return tf.cond(
    #         tf.greater(tf.shape(desc)[0], 1), lambda: AndContext(
    #             self.parse_and(desc[:tf.floordiv(tf.shape(desc)[0], 2), :]),
    #             self.parse_and(desc[tf.floordiv(tf.shape(desc)[0], 2):, :]),
    #             self.comp_fn).get_context(), lambda: self.parse_single_desc(
    #                 desc[0, :]))
    # return tf.map_fn(lambda x: tf.squeeze(self.parse_multiple_descs(x)),
    #                  descs,
    #                  dtype=tf.float32)

    def parse_descs(self, descs, apply=True):
        descs = tf.squeeze(
            tf.map_fn(lambda x: self.parse_single_desc(x),
                      descs,
                      dtype=tf.float32))
        # return descs
        return tf.reshape(self.parse_and(descs), [-1, 8])
        # return tf.map_fn(lambda x: tf.squeeze(
        #     tf.py_function(func=self.parse_and, inp=(x,), Tout=tf.float32)),
        #                  descs,
        #                  dtype=tf.float32)
        # return tf.cond(tf.greater(tf.shape(descs)[1], 1), lambda)
        # if apply:
        #     return self.parse_apply(descs)
        # else:
        #     return self.parse_and(descs)
        # return tf.map_fn(
        #     lambda x: tf.squeeze(self.parse_multiple_descs(x)),
        #     descs,
        #     dtype=tf.float32)


class BasicParser:

    def __init__(self):
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2",
                               trainable=True)
        self.dense1 = tf.layers.Dense(8)

    def parse_descs(self, descs, apply=True):
        embeddings = self.elmo(descs, signature="default",
                               as_dict=True)["default"]
        x = self.dense1(embeddings)
        return x


class GloveParser:

    def __init__(self, use_glove=True):

        self.scope = tf.get_variable_scope()
        if use_glove:
            embedding = np.load('embeddings.npy')

            self.embedding = tf.get_variable('embeddings',
                                             initializer=embedding.astype(
                                                 np.float32),
                                             dtype=tf.float32)
        else:
            self.embedding = tf.get_variable(
                'embeddings', [15, 8], initializer=tf.random_normal_initializer)
        self.dense1 = tf.layers.Dense(8, name='glove_dense')

    def parse_descs(self, descs, apply=True):
        with tf.variable_scope(self.scope):
            embeddings = tf.reduce_sum(
                tf.nn.embedding_lookup(self.embedding, descs), [0, 1])
            # embeddings = tf.reduce_sum(embeddings, [1, 2])
            # embeddings = self.elmo(descs, signature="default",
            #                        as_dict=True)["default"]
            x = tf.reshape(embeddings, [1, -1])
            x = self.dense1(x)
            return x
