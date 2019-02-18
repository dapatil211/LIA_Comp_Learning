from __future__  import absolute_import, division, print_function

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
            return getter(name, shape, dtype, initializer, regularizer, reuse, 
                          trainable, collections, caching_device, 
                          validate_shape, use_resource, constraint, 
                          **kwargs)
        
        dtype = tf.dtypes.as_dtype(dtype)
        shape = tf.TensorShape(shape)
        
        scope = tf.get_variable_scope().name
        name = name + '/' + scope if scope else name
        
        if name in self.vars:
            # Here we handle the case when returning an existing variable.
            if reuse is False:
                tb = self.vars[name].op.traceback[::-1]
                # Throw away internal TF entries and only take a few lines.
                tb = [x for x in tb if 'tensorflow/python' not in x[0]][:3]
                raise ValueError('Variable %s already exists, disallowed.'
                                 ' Did you mean to set reuse=True or '
                                 'reuse=tf.AUTO_REUSE in VarScope? '
                                 'Originally defined at:\n\n%s' % (
                                     name, ''.join(traceback.format_list(tb))))
            
            found_var = self.vars[name]
            
            if not shape.is_compatible_with(found_var.get_shape()):
                raise ValueError('Trying to share variable %s, but specified '
                                 'shape %s and found shape %s.' 
                                 % (name, shape, found_var.get_shape()))
            if not dtype.is_compatible_with(found_var.dtype):
                dtype_str = dtype.name
                found_type_str = found_var.dtype.name
                raise ValueError('Trying to share variable %s, but specified '
                                 'dtype %s and found dtype %s.' 
                                 % (name, dtype_str, found_type_str))
            
            return found_var
        
        # Here we handle the case of creating a new variable.
        if reuse is True:
            raise ValueError('Variable %s does not exist, or was not created '
                             'with tf.get_variable(). Did you mean to set '
                             'reuse=tf.AUTO_REUSE in VarScope?' % name)
        
        with tf.variable_scope('cpg/' + name):
            num_params = shape.num_elements()
            params = self._compute_params(getter, num_params, context)
            params = tf.reshape(params, shape)
            self.vars[name] = params
        
        return params
    
    @abc.abstractmethod
    def _compute_params(self, getter, num_params, context):
        raise NotImplementedError


class LinearCPG(CPG):
    def __init__(self):
        super(LinearCPG, self).__init__()
    
    def _compute_params(self, getter, num_params, context):
        with tf.variable_scope('compute_params'):
            scope = tf.get_variable_scope().name
            weights_name = 'weights'
            weights_name = weights_name + '/' + scope if scope else weights_name
            weights = getter(name=weights_name, 
                             shape=[context.shape[-1], num_params],
                             dtype=context.dtype, 
                             initializer=tf.glorot_uniform_initializer())
            return tf.matmul(context, weights)
