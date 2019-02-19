import tensorflow as tf


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


class ContextParser:

    def __init__(self):
        self.shapes = [
            'square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle',
            'semicircle', 'ellipse', 'any']
        self.colors = [
            'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray', 'any']
        self.embeddings = tf.get_variable('embeddings', [17, 8])
        self.dense1 = tf.layers.Dense(16)
        self.dense2 = tf.layers.Dense(8)
        self.apply_dense1 = tf.layers.Dense(16)
        self.apply_dense2 = tf.layers.Dense(8)

    def lookup_fn(self, index):
        return tf.expand_dims(tf.nn.embedding_lookup(self.embeddings, index), 0)

    def comp_fn(self, c1, c2):
        c1 = self.dense1(c1)
        c2 = self.dense1(c2)
        return self.dense2(tf.nn.selu(
            tf.reduce_sum(tf.stack([c1, c2], axis=0), axis=0)))
    
    def apply_fn(self, adj, base):
        concat = tf.concat([adj, base], axis=-1)
        return self.apply_dense2(tf.nn.selu(
            self.apply_dense1(concat)))

    def parse_single_desc(self, desc):
        shape = PrimitiveContext(desc[0], self.lookup_fn)
        color = PrimitiveContext(desc[1] + 1, self.lookup_fn)
        if desc[0].numpy() == -1:
            shape = PrimitiveContext(8, self.lookup_fn)
        if desc[1].numpy() == -1:
            color = PrimitiveContext(16, self.lookup_fn)
        return ApplyContext(color, shape, self.apply_fn)

    def parse_descs(self, descs):
        return tf.map_fn(
            lambda x: tf.squeeze(
                AndContext(
                    self.parse_single_desc(x[0]), self.parse_single_desc(x[1]),
                    self.comp_fn).get_context()),
            descs,
            dtype=tf.float32)
