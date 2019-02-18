from __future__ import absolute_import, division, print_function

import tensorflow as tf

from cpg import LowRankLinearCPG
from data_loader import load_dil
from comp_module import ContextParser

cpg = LowRankLinearCPG(rank=4)
tf.enable_eager_execution()
tfe = tf.contrib.eager

# TODO: One context per batch.
# TODO: Keep predictions in log-prob space.


def model_fn(images, labels, contexts):
    with tf.variable_scope('mode', use_resource=True, 
                           custom_getter=cpg.getter(contexts)):
        x = tf.layers.conv2d(images, 32, 5, (3, 3), activation=tf.nn.leaky_relu, name='conv1')
        x = tf.layers.max_pooling2d(x, 3, 1, name='pool1')
        x = tf.layers.conv2d(x, 16, 3, activation=tf.nn.leaky_relu, name='conv2')
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool2')
        x = tf.layers.flatten(x, 'flatten')
        x = tf.layers.dense(x, 32, activation=tf.nn.leaky_relu, name='dense1')
        logits = tf.layers.dense(x, 2, name='dense2')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    return predictions, loss


def input_fn(folder, is_train):
    inputs, labels, descs = load_dil(folder)
    num_examples = inputs.shape[0]
    # num_examples = 100
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, descs))
    # dataset = dataset.take(100)
    if is_train:
        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.shuffle(num_examples)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(1000)
    # dataset = dataset.map(parse_context)
    return dataset, num_examples


def train():
    effective_batch_size = 32
    log_steps = 10
    val_steps = 1000
    optimizer = tf.train.AdamOptimizer(0.01)
    global_container = tfe.EagerVariableStore()
    with tf.variable_scope('context_parser', use_resource=True):
        with global_container.as_default():
            parser = ContextParser()

    train_dataset, num_train = input_fn('complearn/train', is_train=True)
    val_dataset, num_val = input_fn('complearn/val', is_train=False)
    tc_train = 0.0
    loss_train = 0.0
    num_train = 0
    step = 0
    last_log_step = -1
    last_val_step = -1
    accumulated_step = 0
    accumulated_grads = []
    for image, label, desc in train_dataset:
        num_train += image.shape[0].value
        with tf.GradientTape() as tape:
            with global_container.as_default():
                context = parser.parse_descs(desc)
                predictions, loss = model_fn(image, label, context)
            tc_train += float(tf.reduce_sum(
                tf.cast(tf.equal(predictions, label), tf.float32)))
            loss = tf.reduce_sum(loss)
            loss_train += float(loss)
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
        log_step = step // effective_batch_size
        if last_log_step != log_step and log_step % log_steps == 0:
            print('Step %d\ttrain loss:%f\ttrain accuracy:%f'
                    % (log_step, loss_train / num_train, tc_train / num_train))
            last_log_step = log_step
            loss_train = 0.0
            tc_train = 0.0
            num_train = 0
        if last_val_step != log_step and log_step % val_steps == 0:
            tc_val = 0.0
            loss_val = 0.0
            for image, label, desc in val_dataset:
                context = parser.parse_descs(desc)
                predictions, loss = model_fn(image, label, context)
                tc_val += float(tf.reduce_sum(
                    tf.cast(tf.equal(predictions, label), tf.float32)))
                loss_val += float(tf.reduce_sum(loss))
            print('Step %d\tval loss:%f\tval accuracy:%f'
                    % (log_step, loss_val / num_val, tc_val / num_val))
            last_val_step = log_step
        step += 1


if __name__ == '__main__':
    train()
