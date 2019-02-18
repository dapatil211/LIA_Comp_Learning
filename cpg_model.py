from __future__ import absolute_import, division, print_function

import tensorflow as tf

from cpg import LinearCPG
from data_loader import load_dil
from comp_module import ContextParser

cpg = LinearCPG()
tf.enable_eager_execution()
tfe = tf.contrib.eager

# TODO: One context per batch.
# TODO: Keep predictions in log-prob space.


def model_fn(images, labels, contexts):
    with tf.variable_scope('mode', use_resource=True, 
                           custom_getter=cpg.getter(contexts)):
        x = tf.layers.conv2d(images, 32, 5, (3, 3), activation=tf.nn.leaky_relu, name='conv1')
        x = tf.layers.max_pooling2d(x, 5, 3, name='pool1')
        x = tf.layers.conv2d(x, 16, 3, activation=tf.nn.leaky_relu, name='conv2')
        x = tf.layers.max_pooling2d(x, 5, 2, name='pool1')
        x = tf.layers.flatten(x, 'flatten')
        x = tf.layers.dense(x, 32, activation=tf.nn.leaky_relu, name='dense1')
        logits = tf.layers.dense(x, 2, activation=tf.nn.leaky_relu, name='dense2')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    return predictions, loss


def input_fn(folder):
    inputs, labels, descs = load_dil(folder)
    num_examples = inputs.shape[0]
    num_examples = 10
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, descs))
    dataset = dataset.take(10)
    dataset = dataset.cache()
    dataset = dataset.shuffle(num_examples)
    dataset = dataset.batch(1)
    # dataset = dataset.map(parse_context)
    return dataset, num_examples


def train():
    EPOCHS = 100
    effective_batch_size = 512
    optimizer = tf.train.AdamOptimizer(0.01)
    global_container = tfe.EagerVariableStore()
    with tf.variable_scope('context_parser', use_resource=True):
        with global_container.as_default():
            parser = ContextParser()

    for epoch in range(EPOCHS):
        train_dataset, num_train = input_fn('complearn/train')
        val_dataset, num_val = input_fn('complearn/val')
        tc_train = 0.0
        tc_val = 0.0
        loss_train = 0.0
        loss_val = 0.0
        accumulated_step = 0
        accumulated_grads = []
        print('Start epoch %d' % epoch)
        for image, label, desc in train_dataset:
            with tf.GradientTape() as tape:
                with global_container.as_default():
                    context = parser.parse_descs(desc)
                    predictions, loss = model_fn(image, label, context)
                tc_train += tf.reduce_sum(
                    tf.cast(tf.equal(predictions, label), tf.float32))
                loss = tf.reduce_sum(loss)
                loss_train += loss
            trainable_vars = global_container.trainable_variables()
            grads = tape.gradient(loss, trainable_vars)
            if accumulated_step < effective_batch_size:
                accumulated_grads.append(grads)
                accumulated_step += 1
            else:
                grads = zip(*accumulated_grads)
                grads = [tf.reduce_sum(tf.stack(g, axis=-1), axis=-1) 
                         for g in grads]
                optimizer.apply_gradients(
                    zip(grads, trainable_vars),
                    global_step=tf.train.get_or_create_global_step())
                accumulated_step = 0
                accumulated_grads = []
        if accumulated_step > 0:
            grads = zip(*accumulated_grads)
            grads = [tf.reduce_sum(tf.stack(g, axis=-1), axis=-1) 
                        for g in grads]
            optimizer.apply_gradients(
                zip(grads, trainable_vars),
                global_step=tf.train.get_or_create_global_step())
            accumulated_step = 0
            accumulated_grads = []
        for image, label, desc in val_dataset:
            context = parser.parse_descs(desc)
            predictions, loss = model_fn(image, label, context)
            tc_val += tf.reduce_sum(
                tf.cast(tf.equal(predictions, label), tf.float32))
            loss_val += tf.reduce_sum(loss)

        print(
            'Epoch %d\ttrain loss:%f\tval loss:%f\ttrain accuracy:%f\tval accuracy:%f\t'
            % (epoch, loss_train / num_train, loss_val / num_val,
               tc_train / num_train, tc_val / num_val))


if __name__ == '__main__':
    train()
