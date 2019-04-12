from __future__ import absolute_import, division, print_function
import argparse
import tensorflow as tf
import os
import random
from comp_module import CompositionalParser, BasicParser
# from cpg import LinearCPG
from graph_cpg import LinearCPG, LowRankLinearCPG
from data_loader import load_dil, load_all_data, create_dataset, create_input_parser
from pprint import pprint
cpg = LinearCPG()
# cpg = LowRankLinearCPG(4)
# tf.enable_eager_execution()
tfe = tf.contrib.eager

# TODO: One context per batch.
# TODO: Keep predictions in log-prob space.


def summaries(v, scope=''):
    with tf.name_scope(scope):
        mean = tf.reduce_mean(v)
        m_s = tf.summary.scalar('mean', mean)
        stddev_s = tf.summary.scalar(
            'stddev', tf.sqrt(tf.reduce_mean(tf.square(v - mean))))
        max_s = tf.summary.scalar('max', tf.reduce_max(v))
        min_s = tf.summary.scalar('min', tf.reduce_min(v))
        hist_s = tf.summary.histogram('histogram', v)
        return tf.summary.merge([m_s, stddev_s, max_s, min_s, hist_s])


def image_summaries(x, scope=''):
    # with tf.name_scope(scope):
    im_summaries = []
    for i in range(x.shape[-1]):
        im_summaries.append(
            tf.summary.image(scope, tf.expand_dims(x[:, :, :, i], 3)))
    return tf.summary.merge(im_summaries)


def model_fn(images,
             labels,
             contexts,
             train,
             pool_dropout=.8,
             fc_dropout=.6,
             context_dropout=.4):
    with tf.variable_scope(
            'model',
            use_resource=True,
            custom_getter=cpg.getter(tf.expand_dims(contexts[0, :], 0)),
    ):
        # images = tf.Print(images, [tf.reduce_max(images)])
        if train:
            contexts = tf.nn.dropout(contexts, 1 - context_dropout)
        # x = tf.layers.batch_normalization(images, training=train, name='bn1')
        x = tf.layers.conv2d(
            images, 32, 5, 1, activation=tf.nn.lrn, name='conv1')
        sum_conv1 = image_summaries(x, 'conv1')
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool1')
        # sum_pool1 = image_summaries(x, 'pool1')
        # if train:
        #     x = tf.nn.dropout(x, 1 - pool_dropout)
        # x = tf.layers.batch_normalization(x, training=train, name='bn2')
        x = tf.layers.conv2d(x, 48, 3, activation=tf.nn.lrn, name='conv2')
        sum_conv2 = image_summaries(x, 'conv2')
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool2')
        # sum_pool2 = image_summaries(x, 'pool2')

        # if train:
        # x = tf.nn.dropout(x, 1 - pool_dropout)
        x = tf.layers.conv2d(x, 64, 3, activation=tf.nn.lrn, name='conv3')
        sum_conv3 = image_summaries(x, 'conv3')
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool3')
        x = tf.layers.conv2d(x, 128, 3, activation=tf.nn.lrn, name='conv4')
        sum_conv4 = image_summaries(x, 'conv4')
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool4')
        # sum_pool2 = image_summaries(x, 'pool4')
        x = tf.layers.flatten(x, 'flatten')
        # if train:
        #     x = tf.nn.dropout(x, keep_prob=1 - fc_dropout)
        x = tf.layers.dense(x, 128, activation=tf.nn.selu, name='dense1')
        logits = tf.layers.dense(x, 2, name='dense2')

        # if train:
        #     contexts = tf.nn.dropout(contexts, 1 - context_dropout)
        # # x = tf.layers.batch_normalization(images, training=train, name='bn1')
        # x = tf.layers.conv2d(
        #     images, 32, 5, (2, 2), activation=tf.nn.lrn, name='conv1', padding='SAME')
        # sum_conv1 = image_summaries(x, 'conv1')
        # x = tf.layers.max_pooling2d(x, 3, 2, name='pool1')
        # sum_pool1 = image_summaries(x, 'pool1')
        # if train:
        #     x = tf.nn.dropout(x, 1 - pool_dropout)
        # # x = tf.layers.batch_normalization(x, training=train, name='bn2')
        # x = tf.layers.conv2d(
        #     x, 16, 3, activation=tf.nn.leaky_relu, name='conv2')
        # sum_conv2 = image_summaries(x, 'conv2')
        # x = tf.layers.max_pooling2d(x, 3, 2, name='pool2')
        # sum_pool2 = image_summaries(x, 'pool2')
        # if train:
        #     x = tf.nn.dropout(x, 1 - pool_dropout)
        # x = tf.layers.flatten(x, 'flatten')
        # # if train:
        # #     x = tf.nn.dropout(x, keep_prob=1 - fc_dropout)
        # x = tf.layers.dense(x, 32, activation=tf.nn.selu, name='dense1')
        # logits = tf.layers.dense(x, 2, name='dense2')
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='ce_loss')
    l2_loss = tf.add_n([
        tf.nn.l2_loss(v)
        for v in tf.trainable_variables()
        if 'model' in v.name and 'kernel' in v.name
    ],
                       name='l2_loss') * .025
    loss = ce_loss + l2_loss
    predictions = tf.argmax(
        logits, axis=1, output_type=tf.int64, name='predictions')
    norm_logits = tf.nn.log_softmax(logits, axis=-1)
    pairs = tf.reshape(norm_logits, [-1, 2, 2])[:, :, 1]
    val_preds = tf.argmax(pairs, axis=1, output_type=tf.int64, name='val_preds')
    val_acc = tf.reduce_mean(
        tf.cast(tf.equal(val_preds, 0), tf.float32), name='val_accuracy')
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, labels), tf.float32), name='accuracy')

    # with tf.name_scope('train' if train else 'val'):
    logits_summary = summaries(logits, 'logits')
    ce_loss_summary = summaries(loss, 'ce_loss')
    with tf.name_scope('l2_loss'):
        l2_loss_summary = tf.summary.scalar('l2_loss', l2_loss)
    loss_summary = summaries(loss, 'loss')
    with tf.name_scope('accuracy'):
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    with tf.name_scope('comparative_accuracy'):
        comp_accuracy_summary = tf.summary.scalar('comparative_accuracy',
                                                  val_acc)

    # all_summaries = tf.summary.merge([
    #     logits_summary, ce_loss_summary, l2_loss_summary, loss_summary,
    #     accuracy_summary
    # ])
    all_summaries = tf.summary.merge([
        logits_summary, ce_loss_summary, l2_loss_summary, loss_summary,
        accuracy_summary, comp_accuracy_summary
    ])
    im_summaries = tf.summary.merge(
        [sum_conv1, sum_conv2, sum_conv3, sum_conv4])
    return loss, accuracy, val_acc, all_summaries, im_summaries


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


file_to_num_examples = {
    'apply/train': 12032,
    'apply/val': 512,
    'apply/test': 1792,
    'and/train': 682240,
    'and/val': 39936,
    'and/test': 80640,
}


def in_fn(file, num_examples, is_train=True, comp=True):
    dataset = create_dataset(file)
    dataset = dataset.map(create_input_parser(comp=comp))
    dataset = dataset.batch(32)
    if is_train:
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(
                min(num_examples // 32, 376)))
    dataset.prefetch(1024)
    return dataset, num_examples


def train(folder='data', name='comp', dataset='apply', summary_dir='summary'):
    # with tf.device('cpu'):
    train_dataset, num_train = in_fn(
        os.path.join(folder, 'train',
                     'dataset.tfrecord'), file_to_num_examples[os.path.join(
                         dataset, 'train')], True, name == 'comp')
    val_dataset, num_val = in_fn(
        os.path.join(folder, 'val',
                     'dataset.tfrecord'), file_to_num_examples[os.path.join(
                         dataset, 'val')], False, name == 'comp')
    test_dataset, num_test = in_fn(
        os.path.join(folder, 'test',
                     'dataset.tfrecord'), file_to_num_examples[os.path.join(
                         dataset, 'test')], False, name == 'comp')
    with tf.variable_scope('context_parser', use_resource=True):
        if name == 'bl':
            parser = BasicParser()
        else:
            parser = CompositionalParser()

    train_iterator = train_dataset.make_initializable_iterator()
    train_image, train_label, train_desc = train_iterator.get_next()
    val_iterator = val_dataset.make_initializable_iterator()
    val_image, val_label, val_desc = val_iterator.get_next()
    optimizer = tf.train.AdamOptimizer(0.00001)

    with tf.variable_scope('model'):
        train_context = parser.parse_descs(train_desc, dataset == 'apply')
        train_loss, train_accuracy, _, train_summary, _ = model_fn(
            train_image, train_label, train_context, True)
    train_loss = tf.reduce_mean(train_loss)
    # train_accuracy = tf.reduce_mean(
    #     tf.cast(tf.equal(train_predictions, train_label), tf.float32))

    # print_op = tf.print({'loss': train_loss, 'acc': train_accuracy}),
    # with tf.control_dependencies(print_op):
    grads = optimizer.compute_gradients(train_loss)
    grad_summaries = []
    for g, v in grads:
        grad_summaries.append(summaries(g, v.name[:-2] + '_grads'))
    context_summary = summaries(train_context, 'context')

    train_summary = tf.summary.merge(grad_summaries +
                                     [train_summary, context_summary])
    if name == 'comp':
        color_summary = summaries(parser.color_embeddings, 'color_embeddings')
        shape_summary = summaries(parser.shape_embeddings, 'shape_embeddings')
        train_summary = tf.summary.merge(
            [train_summary, color_summary, shape_summary])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.apply_gradients(grads)
    train_op = tf.group([train_op, update_ops])
    # train_op = optimizer.minimize(train_loss)
    with tf.variable_scope('model', reuse=True):
        val_context = parser.parse_descs(val_desc, dataset == 'apply')
        val_loss, val_accuracy, val_comp_accuracy, val_summary, val_image_summary = model_fn(
            val_image, val_label, val_context, False)
    val_loss = tf.reduce_mean(val_loss)
    # val_accuracy = tf.reduce_sum(
    #     tf.cast(tf.equal(val_predictions, val_label), tf.float32))
    # pprint(
    # n.name
    # for n in tf.get_default_graph().as_graph_def().node
    # if n.name.startswith('cpg')
    # tf.trainable_variables())
    init = [
        tf.global_variables_initializer(), train_iterator.initializer,
        tf.tables_initializer()
    ]
    step = 0
    cum_train_accuracy = 0.0
    cum_train_loss = 0.0
    log_steps = 10
    train_epoch_steps = max(num_train // 256, 50)
    val_epoch_steps = num_val // 256
    total_val_steps = 0
    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(
            os.path.join(summary_dir) + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(summary_dir) + '/test')
        while True:
            _, batch_loss, batch_acc, summary = sess.run(
                [train_op, train_loss, train_accuracy, train_summary])
            cum_train_accuracy += batch_acc
            cum_train_loss += batch_loss
            train_writer.add_summary(summary, step)
            step += 1
            if step % log_steps == 0:
                print('Step %d\ttrain loss:%f\ttrain accuracy:%f' %
                      (step, cum_train_loss / log_steps,
                       cum_train_accuracy / log_steps))
                cum_train_loss = 0.0
                cum_train_accuracy = 0.0

            if step % train_epoch_steps == 0:
                tc_val = 0.0
                tc_comp_val = 0.0
                loss_val = 0.0
                sess.run(val_iterator.initializer)
                batches = 0.0
                image_viz_step = random.randint(0, val_epoch_steps - 1)
                while True:
                    try:
                        if batches == 0.0:
                            val_batch_loss, val_batch_accuracy, val_batch_comp_accuracy, summary, im_summary = sess.run(
                                [
                                    val_loss, val_accuracy, val_comp_accuracy,
                                    val_summary, val_image_summary
                                ])
                            test_writer.add_summary(im_summary, total_val_steps)
                        else:
                            val_batch_loss, val_batch_accuracy, val_batch_comp_accuracy, summary = sess.run(
                                [
                                    val_loss, val_accuracy, val_comp_accuracy,
                                    val_summary
                                ])
                        tc_val += val_batch_accuracy
                        tc_comp_val += val_batch_comp_accuracy
                        loss_val += val_batch_loss
                        batches += 1.0
                        test_writer.add_summary(summary, total_val_steps)
                        total_val_steps += 1
                    except:
                        print(
                            'Step %d\tval loss:%f\tval accuracy:%f\tval comp accuracy: %f'
                            % (step, loss_val / batches, tc_val / batches,
                               tc_comp_val / batches))
                        break


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
    val_iterator = val_dataset.make_initializable_iterator()
    val_image, val_label, val_desc = val_iterator.get_next()
    optimizer = tf.train.AdamOptimizer(0.001)

    train_context = parser.parse_descs(train_desc)
    train_context = tf.nn.dropout(train_context, .5)

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
                sess.run(val_iterator.initializer)
                while True:
                    try:
                        val_batch_loss, val_batch_accuracy = sess.run(
                            [val_loss, val_accuracy])
                        tc_val += val_batch_accuracy
                        loss_val += val_batch_loss
                    except:
                        print('Step %d\tval loss:%f\tval accuracy:%f' %
                              (step, loss_val / num_val, tc_val / num_val))
                        break


def train_eager(name='comp'):
    tf.enable_eager_execution()

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
                context = tf.nn.dropout(context, .5)
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
    parser.add_argument('-f', '--folder', default='data/')
    parser.add_argument(
        '-d', '--dataset', choices=['apply', 'and'], default='apply')
    parser.add_argument('-s', '--summary', default='summary')

    args = parser.parse_args()
    train(args.folder, args.model, args.dataset, args.summary)
    # if args.model == 'bl':
    #     train_graph(args.model)
    # else:
    #     train_eager(args.model)


if __name__ == '__main__':
    main()