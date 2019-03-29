from __future__ import absolute_import, division, print_function

import abc
import six
import tensorflow as tf
import traceback

from functools import partial


@six.add_metaclass(abc.ABCMeta)
class CPG(object):

    def __init__(self):
        self.vars = dict()

    def getter(self, context):
        return partial(self._getter, context=context)

    def _getter(self,
                context,
                getter,
                name,
                shape=None,
                dtype=tf.float32,
                initializer=None,
                regularizer=None,
                reuse=None,
                trainable=None,
                collections=None,
                caching_device=None,
                partitioner=None,
                validate_shape=True,
                use_resource=None,
                constraint=None,
                **kwargs):
        # If the requested variable is not trainable, just invoke the wrapped
        # variable getter directly.
        if not trainable:
            return getter(
                name=name,
                shape=shape,
                dtype=dtype,
                initializer=initializer,
                regularizer=regularizer,
                reuse=reuse,
                trainable=trainable,
                collections=collections,
                caching_device=caching_device,
                validate_shape=validate_shape,
                use_resource=use_resource,
                constraint=constraint,
                **kwargs)

        dtype = tf.as_dtype(dtype)
        shape = tf.TensorShape(shape)

        scope = tf.get_variable_scope().name
        name = name + '/' + scope if scope else name
        if name in self.vars:
            # Here we handle the case when returning an existing variable.
            # if reuse is False:
            #     tb = self.vars[name].op.traceback[::-1]
            #     # Throw away internal TF entries and only take a few lines.
            #     tb = [x for x in tb if 'tensorflow/python' not in x[0]][:3]
            #     raise ValueError('Variable %s already exists, disallowed.'
            #                     ' Did you mean to set reuse=True or '
            #                     'reuse=tf.AUTO_REUSE in VarScope? '
            #                     'Originally defined at:\n\n%s' % (
            #                         name, ''.join(traceback.format_list(tb))))

            # num_params = shape.num_elements()
            found_vars = self.vars[name]
            # params, _ = self._compute_params(
            #     getter, num_params, context, found_vars)
            # params = tf.reshape(params, shape)

            if not shape.is_compatible_with(found_vars.get_shape()):
                raise ValueError('Trying to share variable %s, but specified '
                                 'shape %s and found shape %s.' %
                                 (name, shape, found_vars.get_shape()))
            if not dtype.is_compatible_with(found_vars.dtype):
                dtype_str = dtype.name
                found_type_str = found_vars.dtype.name
                raise ValueError('Trying to share variable %s, but specified '
                                 'dtype %s and found dtype %s.' %
                                 (name, dtype_str, found_type_str))

            return found_vars

        # Here we handle the case of creating a new variable.
        if reuse is True:
            raise ValueError('Variable %s does not exist, or was not created '
                             'with tf.get_variable(). Did you mean to set '
                             'reuse=tf.AUTO_REUSE in VarScope?' % name)

        with tf.variable_scope('cpg/' + name, use_resource=True):
            num_params = shape.num_elements()
            params = self._compute_params(getter, num_params, context)
            params = tf.reshape(params, shape)
            self.vars[name] = params

        return params

    @abc.abstractmethod
    def _compute_params(self, getter, num_params, context, compute_vars=None):
        raise NotImplementedError


class LinearCPG(CPG):

    def __init__(self):
        super(LinearCPG, self).__init__()

    def _compute_params(self, getter, num_params, context, compute_vars=None):
        with tf.variable_scope('compute_params', use_resource=True):
            scope = tf.get_variable_scope().name
            weights_name = 'weights'
            weights_name = weights_name + '/' + scope if scope else weights_name
            # if compute_vars is not None and weights_name in compute_vars:
            #     weights = compute_vars[weights_name]
            # else:
            weights = getter(
                name=weights_name,
                shape=[context.shape[-1], num_params],
                dtype=context.dtype,
                initializer=tf.glorot_uniform_initializer(),
                use_resource=True)
            return tf.matmul(context, weights)  #, {weights_name: weights}


class LowRankLinearCPG(CPG):

    def __init__(self, rank):
        super(LowRankLinearCPG, self).__init__()
        self.rank = rank

    def _compute_params(self, getter, num_params, context, compute_vars=None):
        with tf.variable_scope('compute_params', use_resource=True):
            scope = tf.get_variable_scope().name
            weights_1_name = 'weights_1'
            weights_1_name = weights_1_name + '/' + scope if scope else weights_1_name
            weights_2_name = 'weights_2'
            weights_2_name = weights_2_name + '/' + scope if scope else weights_2_name
            if compute_vars is not None and weights_1_name in compute_vars:
                weights_1 = compute_vars[weights_1_name]
            else:
                weights_1 = getter(
                    name=weights_1_name,
                    shape=[context.shape[-1], self.rank],
                    dtype=context.dtype,
                    initializer=tf.glorot_uniform_initializer(),
                    use_resource=True)
            if compute_vars is not None and weights_2_name in compute_vars:
                weights_2 = compute_vars[weights_2_name]
            else:
                weights_2 = getter(
                    name=weights_2_name,
                    shape=[self.rank, num_params],
                    dtype=context.dtype,
                    initializer=tf.glorot_uniform_initializer(),
                    use_resource=True)
            return tf.matmul(tf.matmul(context, weights_1), weights_2)  #, \
            #{weights_1_name: weights_1, weights_2_name: weights_2}
