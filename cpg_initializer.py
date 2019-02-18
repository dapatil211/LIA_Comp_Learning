import tensorflow as tf
from six.moves import reduce


class CPGInitializer:

    def __init__(self):
        self.generation_params = {}

    def __call__(self, shape, context, name, dtype=None, partition_info=None):
        variable_name = tf.get_variable_scope().name + '/' + name
        if not variable_name in self.generation_params:
            num_params = reduce((lambda x, y: x * y), shape)
            init_fn = lambda shape, dtype=tf.float32, partition_info=None: tf.glorot_uniform_initializer(
            )(shape, dtype)
            param_generator = tf.get_variable(
                name + '_generator', [context.shape[-1].value, num_params],
                initializer=init_fn)
            self.generation_params[variable_name] = shape, param_generator

        param_shape, param_generator = self.generation_params[variable_name]
        print(name)
        print(context)
        if shape != param_shape:
            raise RuntimeError('Shape does not match previously stored shape')

        if tf.rank(context) < 2:
            context = tf.expand_dims(context, 0)
        return tf.reshape(tf.matmul(context, param_generator), shape)

    def getter(self, context, getter, name, shape, *args, **kwargs):
        if not name in self.generation_params:
            num_params = reduce((lambda x, y: x * y), shape)
            init_fn = lambda shape, dtype=tf.float32, partition_info=None: tf.glorot_uniform_initializer(
            )(shape, dtype)
            param_generator = getter(
                name + '_generator', [context.shape[-1].value, num_params],
                initializer=init_fn)
            self.generation_params[variable_name] = shape, param_generator
        param_shape, param_generator = self.generation_params[variable_name]
        print(name)
        print(context)
        if shape != param_shape:
            raise RuntimeError('Shape does not match previously stored shape')

        if tf.rank(context) < 2:
            context = tf.expand_dims(context, 0)
        return getter(
            name,
            shape,
            initializer=tf.reshape(tf.matmul(context, param_generator), shape))

    def get_config():
        return {}