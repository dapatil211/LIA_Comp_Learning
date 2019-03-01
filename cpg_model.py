from __future__ import absolute_import, division, print_function
import argparse
import tensorflow as tf

from comp_module import CompositionalParser, BasicParser
# from cpg import LinearCPG
from graph_cpg import LinearCPG
from data_loader import load_dil, load_all_data

cpg = LinearCPG()
# tf.enable_eager_execution()
tfe = tf.contrib.eager

# TODO: One context per batch.
# TODO: Keep predictions in log-prob space.


def model_fn(images, labels, contexts, train):
    with tf.variable_scope(
            'model', use_resource=True, custom_getter=cpg.getter(contexts)):
        x = tf.layers.conv2d(
            images, 32, 5, (3, 3), activation=tf.nn.leaky_relu, name='conv1')
        x = tf.layers.max_pooling2d(x, 3, 1, name='pool1')
        x = tf.layers.conv2d(
            x, 16, 3, activation=tf.nn.leaky_relu, name='conv2')
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool2')
        x = tf.layers.flatten(x, 'flatten')
        # if train:
        #     x = tf.nn.dropout(x, keep_prob=.5)
        x = tf.layers.dense(x, 32, activation=tf.nn.selu, name='dense1')
        logits = tf.layers.dense(x, 2, name='dense2')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    return predictions, loss


def input_fn(folder, is_train, is_baseline=False):
    if is_baseline:
        descs, _, _, inputs, labels = load_all_data(folder)
    else:
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


def train_graph(name='comp'):
    is_baseline = 'bl' in name
    log_steps = 10
    effective_batch_size = 32
    val_steps = 500
    train_dataset, num_train = input_fn(
        'complearn/train', is_train=True, is_baseline=is_baseline)
    val_dataset, num_val = input_fn(
        'complearn/val', is_train=False, is_baseline=is_baseline)
    with tf.variable_scope('context_parser', use_resource=True):
        if is_baseline:
            parser = BasicParser()
        else:
            parser = CompositionalParser()
    train_image, train_label, train_desc = train_dataset.make_one_shot_iterator(
    ).get_next()
    val_image, val_label, val_desc = val_dataset.make_one_shot_iterator(
    ).get_next()
    optimizer = tf.train.AdamOptimizer(0.001)

    train_context = parser.parse_descs(train_desc)
    train_predictions, train_loss = model_fn(train_image, train_label,
                                             train_context, True)
    train_loss = tf.reduce_sum(train_loss)
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        val_context = parser.parse_descs(val_desc)
        val_predictions, val_loss = model_fn(val_image, val_label, val_context,
                                             True)
    val_loss = tf.reduce_sum(val_loss)
    val_accuracy = tf.cast(tf.equal(val_predictions, val_label), tf.float32)

    tvs = tf.trainable_variables()
    accumulators = [
        tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
        for tv in tvs
    ]
    accumulation_counter = tf.Variable(0.0, trainable=False)
    loss_counter = tf.Variable(0.0, trainable=False)
    accuracy_counter = tf.Variable(0.0, trainable=False)
    grads = optimizer.compute_gradients(train_loss)
    accumulate_ops = [
        accumulator.assign_add(grad)
        for (accumulator, (grad, var)) in zip(accumulators, grads)
    ]
    accumulate_ops.append(accumulation_counter.assign_add(1.0))
    accumulate_ops.append(loss_counter.assign_add(train_loss))
    accumulate_ops.append(
        accuracy_counter.assign_add(
            tf.reduce_mean(
                tf.cast(tf.equal(train_predictions, train_label), tf.float32))))

    train_op = optimizer.apply_gradients(
        [(accumulator / accumulation_counter, var) \
            for (accumulator, (grad, var)) in zip(accumulators, grads)]
    )
    zero_ops = [
        accumulator.assign(tf.zeros_like(tv))
        for (accumulator, tv) in zip(accumulators, tvs)
    ]
    zero_ops.append(accumulation_counter.assign(0.0))
    zero_ops.append(loss_counter.assign(0.0))
    zero_ops.append(accuracy_counter.assign(0.0))

    init = tf.global_variables_initializer()

    step = 0
    train_accuracy = 0.0
    train_loss = 0.0

    with tf.Session() as sess:
        sess.run(init)
        while True:
            sess.run(zero_ops)
            for i in range(effective_batch_size):
                sess.run(accumulate_ops)
            sess.run(train_op)
            batch_size, train_batch_loss, train_batch_accuracy = sess.run(
                [accumulation_counter, loss_counter, accuracy_counter])
            train_accuracy += train_batch_accuracy / batch_size
            train_loss += train_batch_loss / batch_size
            step += 1
            if step % log_steps == 0:
                print(
                    'Step %d\ttrain loss:%f\ttrain accuracy:%f' %
                    (step, train_loss / log_steps, train_accuracy / log_steps))
                train_loss = 0.0
                train_accuracy = 0.0
            if step % val_steps == 0:
                tc_val = 0.0
                loss_val = 0.0
                while True:
                    try:
                        val_batch_loss, val_batch_accuracy = sess.run(
                            [val_loss, val_accuracy])
                        tc_val += val_batch_accuracy
                        loss_val += val_batch_loss
                    except:
                        print('Step %d\tval loss:%f\tval accuracy:%f' %
                              (step, loss_val / num_val, tc_val / num_val))


def train(name='comp'):
    is_baseline = 'bl' in name
    effective_batch_size = 32
    log_steps = 10
    val_steps = 500
    optimizer = tf.train.AdamOptimizer(0.001)
    global_container = tfe.EagerVariableStore()
    with tf.variable_scope('context_parser', use_resource=True):
        with global_container.as_default():
            if is_baseline:
                parser = BasicParser()
            else:
                parser = CompositionalParser()

    train_dataset, num_train = input_fn(
        'complearn/train', is_train=True, is_baseline=is_baseline)
    val_dataset, num_val = input_fn(
        'complearn/val', is_train=False, is_baseline=is_baseline)
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
                predictions, loss = model_fn(image, label, context, True)
            tc_train += float(
                tf.reduce_sum(
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
            grads = [
                tf.reduce_sum(tf.stack(g, axis=-1), axis=-1) for g in grads
            ]
            optimizer.apply_gradients(
                zip(grads, trainable_vars),
                global_step=tf.train.get_or_create_global_step())
            accumulated_step = 0
            accumulated_grads = []
        log_step = step // effective_batch_size
        if last_log_step != log_step and log_step % log_steps == 0:
            print('Step %d\ttrain loss:%f\ttrain accuracy:%f' %
                  (log_step, loss_train / num_train, tc_train / num_train))
            last_log_step = log_step
            loss_train = 0.0
            tc_train = 0.0
            num_train = 0
        if last_val_step != log_step and log_step % val_steps == 0:
            tc_val = 0.0
            loss_val = 0.0
            for image, label, desc in val_dataset:
                context = parser.parse_descs(desc)
                predictions, loss = model_fn(image, label, context, False)
                tc_val += float(
                    tf.reduce_sum(
                        tf.cast(tf.equal(predictions, label), tf.float32)))
                loss_val += float(tf.reduce_sum(loss))
            print('Step %d\tval loss:%f\tval accuracy:%f' %
                  (log_step, loss_val / num_val, tc_val / num_val))
            last_val_step = log_step
        step += 1


def main():
    parser = argparse.ArgumentParser(description='Compositional Learning')
    parser.add_argument('-m', '--model', choices=['bl', 'comp'], default='comp')
    args = parser.parse_args()
    train_graph(args.model)


if __name__ == '__main__':
    main()