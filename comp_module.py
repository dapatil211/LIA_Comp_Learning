import tensorflow as tf


class PrimitiveContext:

    def __init__(self, index, lookup_fn):
        self.index = index
        self.lookup_fn = lookup_fn

    def get_context(self):
        return self.lookup_fn(self.index)


class AndContext:

    def __init__(self, c1, c2, comp_fn):
        self.c1 = c1
        self.c2 = c2
        self.comp_fn = comp_fn

    def get_context(self):
        return self.comp_fn(self.c1.get_context(), self.c2.get_context())


class ContextParser:

    def __init__(self):
        self.shapes = [
            'square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle',
            'semicircle', 'ellipse'
        ]
        self.colors = [
            'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray'
        ]
        self.embeddings = tf.get_variable('embeddings', [15, 8])
        self.dense1 = tf.layers.Dense(16)
        self.dense2 = tf.layers.Dense(8)

    def lookup_fn(self, index):
        return tf.expand_dims(tf.nn.embedding_lookup(self.embeddings, index), 0)

    def comp_fn(self, c1, c2):
        x = self.dense1(tf.concat([c1, c2], 1))
        return self.dense2(x)

    def parse_single_desc(self, desc):
        if desc[0].numpy() == -1:
            return PrimitiveContext(desc[1], self.lookup_fn)
        elif desc[1].numpy() == -1:
            return PrimitiveContext(desc[0], self.lookup_fn)
        else:
            return AndContext(
                PrimitiveContext(desc[0], self.lookup_fn),
                PrimitiveContext(desc[1], self.lookup_fn), self.comp_fn)

    def parse_descs(self, descs):
        return tf.map_fn(
            lambda x: tf.squeeze(
                AndContext(
                    self.parse_single_desc(x[0]), self.parse_single_desc(x[1]),
                    self.comp_fn).get_context()),
            descs,
            dtype=tf.float32)
