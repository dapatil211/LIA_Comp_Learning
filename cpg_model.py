from __future__ import absolute_import, division, print_function
import argparse
import tensorflow as tf
import numpy as np
import csv
import os
import random
from comp_module import CompositionalParser, BasicParser, GloveParser
# from cpg import LinearCPG
from graph_cpg import LinearCPG, LowRankLinearCPG
from data_loader import load_dil, load_all_data, create_dataset, InputParser
from pprint import pprint, pformat
from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard.plugins.beholder import Beholder
from amsgrad import AMSGrad
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
# logging.getLogger().setLevel(logging.INFO)
# logging.getLogger().propagate = False
tf.logging.set_verbosity(tf.logging.INFO)
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


def weight_summary(weights):
    weight_summaries = []
    for weight in weights:
        weight_summaries.append(tf.summary.image(weight.name, weight))
    return tf.summary.merge(weight_summaries)


def model_fn(images,
             labels,
             contexts,
             train,
             cpg,
             pool_dropout=.8,
             fc_dropout=.6,
             context_dropout=.4,
             l2_weight=.005):
    # contexts = tf.Print(contexts, [contexts[0, :]], summarize=8)
    logits, all_summaries = create_logits(images,
                                          contexts,
                                          train,
                                          cpg,
                                          fc_dropout=.6,
                                          context_dropout=.4,
                                          add_summaries=True)
    # weight_summaries = tf.summary.merge([
    #     weight_summary(weights) for weights in [
    #         conv1.trainable_weights, conv2.trainable_weights,
    #         conv3.trainable_weights, conv4.trainable_weights,
    #         dense1.trainable_weights, dense2.trainable_weights
    #     ]
    # ])
    loss, metrics_dict, update_ops = construct_loss_and_metrics(
        labels, logits, l2_weight, train, all_summaries)
    return loss, metrics_dict, update_ops  #, weight_summaries


def construct_loss_and_metrics(labels, logits, l2_weight, train, all_summaries):
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                             logits=logits,
                                                             name='ce_loss')
    # logits = tf.Print(logits, [logits])
    with tf.name_scope('l2_loss'):

        l2_loss = tf.add_n([
            tf.nn.l2_loss(v)
            for v in tf.trainable_variables()
            if 'model' in v.name and 'kernel' in v.name
        ],
                           name='l2_loss') * l2_weight
    loss = ce_loss + l2_loss
    predictions = tf.argmax(logits,
                            axis=1,
                            output_type=tf.int64,
                            name='predictions')

    norm_logits = tf.nn.log_softmax(logits, axis=-1)
    # norm_logits = tf.Print(norm_logits, [tf.exp(norm_logits)])
    pairs = tf.reshape(norm_logits, [-1, 2, 2])[:, :, 1]
    # pairs = tf.Print(pairs, [pairs])

    val_preds = tf.argmax(pairs, axis=1, output_type=tf.int64, name='val_preds')
    # labels = tf.Print(labels, [labels, predictions], summarize=32)
    metrics_dict = {
        # 'loss':
        # tf.metrics.mean(loss),
        'l2_loss':
        tf.metrics.mean(l2_loss, name='l2_loss'),
        'ce_loss':
        tf.metrics.mean(ce_loss, name='ce_loss'),
        'accuracy':
        tf.metrics.accuracy(labels, predictions, name='accuracy'),
        'comparative_accuracy':
        tf.metrics.accuracy(tf.zeros_like(val_preds),
                            val_preds,
                            name='comparative_accuracy'),
    }
    for k in metrics_dict:
        tf.summary.scalar(k, metrics_dict[k][0])
    update_ops = tf.group([metrics_dict[k][1] for k in metrics_dict])

    if not train:
        with tf.name_scope('predictions'):
            all_summaries.append(
                tf.summary.histogram('predictions', predictions))
            all_summaries.append(
                tf.summary.histogram('comparative predictions', val_preds))

        all_summaries.append(summaries(logits, 'logits'))
    return loss, metrics_dict, update_ops


def create_logits(images,
                  contexts,
                  train,
                  cpg,
                  fc_dropout=.6,
                  context_dropout=.4,
                  add_summaries=True):
    with tf.variable_scope(
            'model',
            use_resource=True,
            custom_getter=cpg.getter(tf.expand_dims(contexts[0, :], 0)),
    ):
        # images = tf.Print(images, [tf.reduce_max(images)])
        im_summaries = []
        all_summaries = []
        if train:
            contexts = tf.nn.dropout(contexts, rate=context_dropout)
        # x = tf.layers.batch_normalization(images, training=train, name='bn1')
        # bn1 = tf.layers.BatchNormalization()
        # conv1 = tf.layers.Conv2D(32, 5, activation=tf.nn.lrn, name='conv1')
        conv1 = tf.layers.Conv2D(64,
                                 3,
                                 activation=tf.nn.leaky_relu,
                                 name='conv1',
                                 padding='SAME')
        # bn2 = tf.layers.BatchNormalization()
        conv2 = tf.layers.Conv2D(64,
                                 3,
                                 activation=tf.nn.leaky_relu,
                                 name='conv2',
                                 padding='SAME')
        # bn3 = tf.layers.BatchNormalization()
        conv3 = tf.layers.Conv2D(128,
                                 3,
                                 activation=tf.nn.leaky_relu,
                                 name='conv3',
                                 padding='SAME')
        # bn4 = tf.layers.BatchNormalization()
        conv4 = tf.layers.Conv2D(256,
                                 3,
                                 activation=tf.nn.leaky_relu,
                                 name='conv4',
                                 padding='SAME')
        conv5 = tf.layers.Conv2D(256,
                                 3,
                                 activation=tf.nn.leaky_relu,
                                 name='conv5',
                                 padding='SAME')
        dense1 = tf.layers.Dense(256, activation=tf.nn.selu, name='dense1')
        dense2 = tf.layers.Dense(2, name='dense2')
        x = images
        # x = bn1(x, training=train)
        x = conv1(x)
        if not train and add_summaries:
            im_summaries.append(image_summaries(x, 'act_conv1'))

        x = tf.layers.max_pooling2d(x, 3, 2, name='pool1')
        # sum_pool1 = image_summaries(x, 'pool1')
        # if train:
        #     x = tf.nn.dropout(x, 1 - pool_dropout)
        # x = tf.layers.batch_normalization(x, training=train, name='bn2')
        # x = bn2(x, training=train)
        x = conv2(x)
        if not train and add_summaries:
            im_summaries.append(image_summaries(x, 'act_conv2'))

        x = tf.layers.max_pooling2d(x, 3, 2, name='pool2')
        # sum_pool2 = image_summaries(x, 'pool2')

        # if train:
        # x = tf.nn.dropout(x, 1 - pool_dropout)
        # x = bn3(x, training=train)
        x = conv3(x)
        if not train and add_summaries:
            im_summaries.append(image_summaries(x, 'act_conv3'))
        x = tf.layers.max_pooling2d(x, 3, 2, name='pool3')
        # x = bn4(x, training=train)
        x = conv4(x)
        x = tf.layers.max_pooling2d(x, 3, 1, name='pool4')
        x = conv5(x)
        # if not train:
        #     im_summaries.append(image_summaries(x, 'act_conv4'))
        # x = tf.layers.max_pooling2d(x, 3, 2, name='pool4')
        # sum_pool2 = image_summaries(x, 'pool4')
        x = tf.reduce_mean(x, axis=[1, 2], name='global_pool')
        if train:
            x = tf.nn.dropout(x, rate=fc_dropout)
        x = dense1(x)
        logits = dense2(x)
    return logits, all_summaries


file_to_num_examples = {
    'apply/train': 12032,
    'apply/val': 512,
    'apply/insample_val': 1504,
    'apply/test': 1792,
    'and/train': 682240,
    'and/val': 39936,
    'and/test': 80640,
    'and/insample_val': 85280,
    'small_conj/train': 9984,
    'small_conj/val': 512,
    'small_conj/test': 1024,
    'small_conj/insample_val': 512,
    'small_conj/insample_test': 1024,
    'large_conj_2/train': 42848,
    'large_conj_2/val': 2112,
    'large_conj_2/test': 4256,
    'large_conj_2/insample_val': 2112,
    'large_conj_2/insample_test': 4256,
    'large_conj/train': 100000,
    'large_conj/val': 4992,
    'large_conj/test': 9984,
    'large_conj/insample_val': 4992,
    'large_conj/insample_test': 9984,
    'mini_conj/train': 1024,
    'mini_conj/val': 4992,
    'mini_conj/test': 9984,
    'mini_conj/insample_val': 4992,
    'mini_conj/insample_test': 9984
}

COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan", "gray"]

SHAPES = [
    'square',
    'rectangle',
    'triangle',
    'pentagon',
    'cross',
    'circle',
    'semicircle',
    'ellipse',
]


def write_metadata(folder, comp):
    with open(os.path.join(folder, 'metadata.tsv'), "w") as record_file:
        captions = generate_possible_captions(comp)
        record_file.write("label\tcolor\tshape\n")
        for caption in captions:
            parts = caption.split()
            if len(parts) == 2:
                color = parts[0]
                shape = parts[1]
            elif parts[0] in SHAPES:
                color = 'none'
                shape = parts[0]
            else:
                color = parts[0]
                shape = 'none'
            record_file.write("%s\t%s\t%s\n" % (caption, color, shape))
    with open(os.path.join(folder, 'color.tsv'), "w") as record_file:
        for color in COLORS:
            record_file.write("%s\n" % color)
        record_file.write("none\n")
    with open(os.path.join(folder, 'shape.tsv'), "w") as record_file:
        for shape in SHAPES:
            record_file.write("%s\n" % shape)
        record_file.write("none\n")


def generate_possible_captions(comp):
    captions = COLORS + SHAPES if comp else []
    for color in COLORS:
        for shape in SHAPES:
            captions.append(color + ' ' + shape)
    return captions


def estimator_in_fn(file, parser, batch_size, train_steps, is_train=True):
    dataset = create_dataset(file, to_dict=True)
    # dataset = dataset.map(parser.parse_to_dict)
    dataset = dataset.batch(batch_size)
    if is_train:
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(min(train_steps, 376)))
    dataset.prefetch(1024)
    return dataset


def get_learning_rate(init_lr, lr_decay_method, total_steps, step):
    if lr_decay_method == 'constant':
        return tf.constant(init_lr)
    elif lr_decay_method == 'linear':
        init_lr = tf.constant(init_lr)
        step = tf.cast(step, tf.float32)
        return init_lr - (init_lr - 1e-6) * (total_steps - step) / total_steps
    elif lr_decay_method == 'exp':
        return tf.train.exponential_decay(init_lr, step, int(total_steps / 1.5),
                                          .1)
    elif lr_decay_method == 'cosine':
        return tf.train.cosine_decay(init_lr, step, total_steps)
    raise ValueError('Optimizer not found')


def get_optimizer(optimizer_name, learning_rate, **kwargs):
    if optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate, **kwargs)
    elif optimizer_name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, **kwargs)
    elif optimizer_name == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, momentum=.9)
    elif optimizer_name == 'adadelta':
        return tf.train.AdadeltaOptimizer(learning_rate, **kwargs)
    elif optimizer_name == 'amsgrad':
        return AMSGrad(learning_rate=learning_rate)
    raise ValueError('Optimizer not found')


def full_model_fn(features, labels, mode, params):
    cpg = LinearCPG()
    input_parser = InputParser(True)
    if params['baseline']:
        parser = GloveParser(use_glove=params['glove'])
    else:
        parser = CompositionalParser(
            hidden_dimension_size=params['comp_hidden_dimension'],
            use_glove=params['glove'])
    features, labels = input_parser.parse_to_dict(features, labels)

    images = features['image']
    descs = features['concept']
    contexts = parser.parse_descs(descs[0], params['dataset'] == 'apply')
    loss, metrics, metric_update_ops = model_fn(
        images,
        labels,
        contexts,
        mode == tf.estimator.ModeKeys.TRAIN,
        cpg,
        pool_dropout=params['pool_dropout'],
        fc_dropout=params['fc_dropout'],
        context_dropout=params['context_dropout'],
        l2_weight=params['l2_weight'],
    )
    learning_rate = get_learning_rate(params['init_lr'],
                                      params['lr_decay_method'],
                                      params['total_steps'],
                                      tf.train.get_or_create_global_step())
    optimizer = get_optimizer(params['optimizer'], learning_rate,
                              **params['optimizer_args'])
    tf.summary.scalar('learning_rate', learning_rate)

    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, global_norm = tf.clip_by_global_norm(gradients, 10.0)

    # grads = optimizer.compute_gradients(loss)
    grads = zip(gradients, variables)
    grad_summaries = []
    tf.summary.scalar('gradient_norm', global_norm)
    new_grads = []
    for g, v in grads:
        if g is not None:
            new_grads.append((g, v))
            grad_summaries.append(summaries(g, v.name[:-2] + '_grads'))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.apply_gradients(
        new_grads, global_step=tf.train.get_or_create_global_step())
    train_op = tf.group([train_op, update_ops, metric_update_ops])
    train_op = train_op if mode == tf.estimator.ModeKeys.TRAIN else None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=tf.reduce_mean(loss),
                                      train_op=train_op,
                                      eval_metric_ops=metrics)

    # (train_loss, train_metrics, train_summary_op, _,
    #  train_metric_ops) = model_fn(train_image, train_label, train_context, True)


def output_comp_model_fn(features, labels, mode, params):
    cpg = LinearCPG()
    input_parser = InputParser(True)
    features, labels = input_parser.parse_to_dict(features, labels)
    images = features['image']
    descs = features['concept']
    parser = GloveParser(use_glove=params['glove'])

    # batch_size = descs.get_shape()[0]
    # log_softmax = tf.constant([[]], shape=(batch_size, 0))
    # shape_idx = tf.constant(0, dtype=tf.int32)

    def create_single_shape_logits(desc):
        context = parser.parse_descs([desc])
        logits, _ = create_logits(
            images,
            context,
            mode == tf.estimator.ModeKeys.TRAIN,
            cpg,
            fc_dropout=params['fc_dropout'],
            context_dropout=params['context_dropout'],
            add_summaries=False,
        )
        return tf.log_sigmoid(logits[:, 0])
        # tf.concat(
        #     1, [log_sigmoid, tf.nn.log_sigmoid(logits[:, 0])])
        # shape_idx = tf.add(shape_idx, 1), all_summaries
        # return image, descs, log_softmax, shape_idx

    labels = tf.cast(labels, tf.float32)
    log_sigmoid = tf.map_fn(lambda x: create_single_shape_logits(x),
                            descs[0],
                            dtype=tf.float32)
    logits = tf.reduce_sum(log_sigmoid, 0)
    # logits = tf.Print(logits, [tf.shape(logits), tf.shape(log_sigmoid)])
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                      labels=labels)
    with tf.name_scope('l2_loss'):
        l2_loss = tf.add_n([
            tf.nn.l2_loss(v)
            for v in tf.trainable_variables()
            if 'model' in v.name and 'kernel' in v.name
        ],
                           name='l2_loss') * params['l2_weight']
    loss = ce_loss + l2_loss
    predictions = logits > 0.0
    pairs = tf.reshape(logits, [-1, 2])
    val_preds = tf.argmax(pairs, axis=1, output_type=tf.int64, name='val_preds')

    metrics = {
        # 'loss':
        # tf.metrics.mean(loss),
        'l2_loss':
        tf.metrics.mean(l2_loss, name='l2_loss'),
        'ce_loss':
        tf.metrics.mean(ce_loss, name='ce_loss'),
        'accuracy':
        tf.metrics.accuracy(labels, predictions, name='accuracy'),
        'comparative_accuracy':
        tf.metrics.accuracy(tf.zeros_like(val_preds),
                            val_preds,
                            name='comparative_accuracy'),
    }
    for k in metrics:
        tf.summary.scalar(k, metrics[k][0])
    metric_update_ops = tf.group([metrics[k][1] for k in metrics])

    # log_sigmoid = tf.while_loop(create_single_shape_logits,

    #     cond=lambda image, descs, log_sigmoid: tf.less(
    #         tf.shape(log_sigmoid)[1],
    #         tf.shape(descs)[1]),
    #     body=create_single_shape_logits,
    #     loop_vars=[images, descs, log_softmax],
    #     shape_invariants=[
    #         images.get_shape(),
    #         descs.get_shape(),
    #         tf.TensorShape([batch_size, 1]),
    #         shape_idx.get_shape()
    #     ], parallel_iterations=1)
    # logits = tf.reduce_sum(log_softmax[:, :, 0], 1)
    # logits = tf.stack([logits, tf.log(1.0 - tf.exp(logits))], 1)
    # softmax = tf.exp(tf.reduce_sum(log_softmax[:, :, 0], 1))
    # labels = tf.cast(labels, tf.float32)
    # loss = -1 * (labels * tf.log(softmax) +
    #              (1.0 - labels) * tf.log(1.0 - softmax))
    # predictions = softmax > .5
    # accuracy
    # if params['baseline']:
    # else:
    #     parser = CompositionalParser(
    #         hidden_dimension_size=params['comp_hidden_dimension'],
    #         use_glove=params['glove'])

    # contexts = parser.parse_descs(descs[0], params['dataset'] == 'apply')
    # loss, metrics, metric_update_ops = model_fn(
    #     images,
    #     labels,
    #     contexts,
    #     mode == tf.estimator.ModeKeys.TRAIN,
    #     cpg,
    #     pool_dropout=params['pool_dropout'],
    #     fc_dropout=params['fc_dropout'],
    #     context_dropout=params['context_dropout'],
    #     l2_weight=params['l2_weight'],
    # )
    learning_rate = get_learning_rate(params['init_lr'],
                                      params['lr_decay_method'],
                                      params['total_steps'],
                                      tf.train.get_or_create_global_step())
    optimizer = get_optimizer(params['optimizer'], learning_rate,
                              **params['optimizer_args'])
    tf.summary.scalar('learning_rate', learning_rate)

    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, global_norm = tf.clip_by_global_norm(gradients, 10.0)

    # grads = optimizer.compute_gradients(loss)
    grads = zip(gradients, variables)
    grad_summaries = []
    tf.summary.scalar('gradient_norm', global_norm)
    new_grads = []
    for g, v in grads:
        if g is not None:
            new_grads.append((g, v))
            grad_summaries.append(summaries(g, v.name[:-2] + '_grads'))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.apply_gradients(
        new_grads, global_step=tf.train.get_or_create_global_step())
    train_op = tf.group([train_op, update_ops, metric_update_ops])
    train_op = train_op if mode == tf.estimator.ModeKeys.TRAIN else None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=tf.reduce_mean(loss),
                                      train_op=train_op,
                                      eval_metric_ops=metrics)


def train_estimator(folder='data',
                    name='comp',
                    dataset='apply',
                    summary_dir='summary',
                    params=None):
    BATCH_SIZE = 32
    if dataset.startswith('large_conj') and dataset != 'large_conj_2':
        dataset = 'large_conj'
    elif dataset.startswith('mini'):
        dataset = 'mini_conj'
    steps_train = file_to_num_examples[os.path.join(dataset,
                                                    'train')] // BATCH_SIZE
    steps_train *= params['epochs_between_evals']
    steps_val = file_to_num_examples[os.path.join(dataset, 'val')] // BATCH_SIZE
    steps_in_val = file_to_num_examples[os.path.join(
        dataset, 'insample_val')] // BATCH_SIZE
    steps_test = file_to_num_examples[os.path.join(dataset,
                                                   'test')] // BATCH_SIZE
    steps_in_test = file_to_num_examples[os.path.join(
        dataset, 'insample_test')] // BATCH_SIZE
    params['baseline'] = name == 'bl'
    params['dataset'] = dataset
    estimator_model_fn = output_comp_model_fn if name == 'comp_output' else full_model_fn
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(save_summary_steps=100,
                                    save_checkpoints_steps=steps_train * 10,
                                    log_step_count_steps=50,
                                    session_config=session_config)
    estimator = tf.estimator.Estimator(model_fn=estimator_model_fn,
                                       model_dir=summary_dir,
                                       params=params,
                                       config=config)
    parser = InputParser(True)
    # steps = 0
    ckpt = tf.train.get_checkpoint_state(summary_dir)

    #Extract from checkpoint filename
    steps = int(os.path.basename(
        ckpt.model_checkpoint_path).split('-')[1]) if ckpt else 0
    tf.logging.info('STEPS: %d', steps)
    test_results = None
    try:
        while steps < params['total_steps']:
            estimator.train(
                lambda: estimator_in_fn(
                    os.path.join(folder, 'train', 'dataset.tfrecord'), parser,
                    BATCH_SIZE, steps_train, True),
                steps=steps_train,
            )
            val_results = estimator.evaluate(
                lambda: estimator_in_fn(
                    os.path.join(folder, 'outsample_val', 'dataset.tfrecord'),
                    parser, BATCH_SIZE, steps_val, False),
                steps=steps_val,
                name='outsample_val',
            )
            in_val_results = estimator.evaluate(
                lambda: estimator_in_fn(
                    os.path.join(folder, 'insample_val', 'dataset.tfrecord'),
                    parser, BATCH_SIZE, steps_in_val, False),
                steps=steps_in_val,
                name='insample_val',
            )
            out_test_results = estimator.evaluate(
                lambda: estimator_in_fn(
                    os.path.join(folder, 'outsample_test', 'dataset.tfrecord'),
                    parser, BATCH_SIZE, steps_test, False),
                steps=steps_test,
                name='outsample_test',
            )
            in_test_results = estimator.evaluate(
                lambda: estimator_in_fn(
                    os.path.join(folder, 'insample_test', 'dataset.tfrecord'),
                    parser, BATCH_SIZE, steps_test, False),
                steps=steps_in_test,
                name='insample_test',
            )
            steps += steps_train
            tf.logging.info('outsample Validation:\n%s' % pformat(val_results))
            tf.logging.info('insample Validation:\n%s' %
                            pformat(in_val_results))
            tf.logging.info('outsample Test:\n%s' % pformat(out_test_results))
            tf.logging.info('insample Test:\n%s' % pformat(in_test_results))
        # out_test_results = estimator.evaluate(
        #     lambda: estimator_in_fn(
        #         os.path.join(folder, 'outsample_test', 'dataset.tfrecord'),
        #         parser, BATCH_SIZE, steps_test, False),
        #     steps=steps_test,
        #     name='outsample_test',
        # )
        # in_test_results = estimator.evaluate(
        #     lambda: estimator_in_fn(
        #         os.path.join(folder, 'insample_test', 'dataset.tfrecord'),
        #         parser, BATCH_SIZE, steps_test, False),
        #     steps=steps_in_test,
        #     name='insample_test',
        # )
        # tf.logging.info('outsample Test:\n%s' % pformat(out_test_results))
        # tf.logging.info('insample Test:\n%s' % pformat(in_test_results))
    except KeyboardInterrupt:
        pass
        # out_test_results = estimator.evaluate(
        #     lambda: estimator_in_fn(
        #         os.path.join(folder, 'outsample_test', 'dataset.tfrecord'),
        #         parser, BATCH_SIZE, steps_test, False),
        #     steps=steps_test,
        #     name='outsample_test',
        # )
        # in_test_results = estimator.evaluate(
        #     lambda: estimator_in_fn(
        #         os.path.join(folder, 'insample_test', 'dataset.tfrecord'),
        #         parser, BATCH_SIZE, steps_test, False),
        #     steps=steps_in_test,
        #     name='insample_test',
        # )
        # tf.logging.info('outsample Test:\n%s' % pformat(out_test_results))
        # tf.logging.info('insample Test:\n%s' % pformat(in_test_results))
    # tf.logging.info('Test:\n%s' % pformat(test_results))

    # metadata_ file = os.path.join(summary_dir, 'train')
    # write_metadata(metadata_file, name == 'comp')


def main():
    parser = argparse.ArgumentParser(description='Compositional Learning')
    parser.add_argument('-m',
                        '--model',
                        choices=['bl', 'comp', 'comp_output'],
                        default='comp')
    parser.add_argument('-f', '--folder', default='data/')
    parser.add_argument('-d', '--dataset', default='apply')
    parser.add_argument('-s', '--summary', default='summary')
    parser.add_argument('--pool-dropout', default=0, type=float)
    parser.add_argument('--fc-dropout', default=.5, type=float)
    parser.add_argument('--context-dropout', default=.2, type=float)
    parser.add_argument('--l2-weight', default=.005, type=float)
    parser.add_argument('--init-lr', default=.01, type=float)
    parser.add_argument('--lr-decay-method',
                        default='linear',
                        choices=['constant', 'linear', 'exp', 'cosine'])
    parser.add_argument('--total-steps', default=80000, type=int)
    parser.add_argument('--epochs-between-evals', default=1, type=int)

    parser.add_argument(
        '--optimizer',
        default='adam',
        choices=['adam', 'rmsprop', 'momentum', 'adadelta', 'amsgrad'])
    parser.add_argument('--comp-hidden-dimension', default=16, type=int)
    parser.add_argument('--glove', action='store_true')

    args = parser.parse_args()
    params = {
        'pool_dropout': args.pool_dropout,
        'fc_dropout': args.fc_dropout,
        'context_dropout': args.context_dropout,
        'l2_weight': args.l2_weight,
        'init_lr': args.init_lr,
        'lr_decay_method': args.lr_decay_method,
        'total_steps': args.total_steps,
        'optimizer': args.optimizer,
        'optimizer_args': {},
        'comp_hidden_dimension': args.comp_hidden_dimension,
        'glove': args.glove,
        'epochs_between_evals': args.epochs_between_evals,
    }

    train_estimator(args.folder, args.model, args.dataset, args.summary, params)
    # if args.model == 'bl':
    #     train_graph(args.model)
    # else:
    #     train_eager(args.model)


if __name__ == '__main__':
    main()