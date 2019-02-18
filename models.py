import tensorflow as tf
import tensorflow_hub as hub
tfd = tf.contrib.distributions

IMAGE_EMBED_SIZE = 1080


def elmo(sentences):
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    embeddings = elmo(sentences, signature="default", as_dict=True)["elmo"]
    return embeddings


def examples_encoder(examples):
    examples = tf.reshape(examples, [-1, IMAGE_EMBED_SIZE])
    examples = tf.layers.dense(examples, 512, activation=tf.nn.relu)
    examples = tf.layers.dense(examples, 256, activation=tf.nn.relu)
    examples = tf.layers.dense(examples, 128, activation=tf.nn.relu)
    examples = tf.reshape(examples, [-1, 4, 128])
    examples = tf.map_fn(
        lambda x: tf.map_fn(
            lambda y: tf.map_fn(lambda z: tf.concat([y, z], 0), x), x),
        examples)
    examples = tf.reshape(examples, [-1, 256])
    examples = tf.layers.dense(examples, 256, activation=tf.nn.relu)
    examples = tf.layers.dense(examples, 128, activation=tf.nn.relu)
    examples = tf.reshape(examples, [-1, 16, 128])
    examples = tf.map_fn(lambda x: tf.reduce_sum(examples, axis=1))
    return examples


def dense(x, weights):
    layer_size = weights.shape()[-1].value
    bias = tf.Variable(tf.zeros([layer_size]))
    return tf.add(tf.matmul(x, weights), bias)


"""
No parameter generator, image examples embedding and text embedding concatenated,
with image embedding.
"""


def baseline_model_1(desc, lens, examples, inputs, labels, batch_size=32):
    embeddings = elmo(desc)
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(128)
    initial_state = rnn_cell.zero_state(tf.shape(lens), dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(
        rnn_cell,
        embeddings,
        sequence_length=lens,
        initial_state=initial_state,
        dtype=tf.float32)

    # example_code = examples_encoder(examples)
    # concept_rep = tf.concat([inputs, example_code, outputs[:, -1, :]], 0)
    concept_rep = tf.concat([inputs, outputs[:, -1, :]], 1)
    x = tf.layers.dense(
        concept_rep, 32, activation=tf.nn.leaky_relu, name='dense1')
    logits = tf.layers.dense(x, 2, name='dense2')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_sum(loss)
    predictions = tf.argmax(logits, axis=1)
    with tf.name_scope('metrics'):
        epoch_loss, epoch_loss_op = tf.metrics.mean(loss)
        acc, acc_op = tf.metrics.accuracy(labels, predictions)
    return logits, loss, acc, acc_op, epoch_loss, epoch_loss_op


"""
Add parameter generator, concept representation constructed using images and text.
No explicitly modeled conjunction module.
Conjunction will be looked at post-hoc
"""


def baseline_model_2(desc, lens, examples, inputs, labels):
    embeddings = elmo(desc)
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(128)
    initial_state = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(
        rnn_cell,
        embeddings,
        sequence_length=lens,
        initial_state=initial_state,
        dtype=tf.float32)

    example_code = examples_encoder(examples)
    concept_rep = tf.concat([example_code, outputs[:, -1, :]], 0)
    concept_rep = tf.layers.dense(concept_rep, 512, activation=tf.nn.relu)
    concept_rep = tf.layers.dense(
        concept_rep, 128 * 2 * 2, activation=tf.nn.relu)
    dist = tfd.MultivariateNormalDiag(concept_rep[:, :128 * 2],
                                      concept_rep[:, 128 * 2:])
    # Conjunction module
    # concept_rep = tf.layers.dense(concept_rep, 512, activation=tf.nn.relu)
    # theta = tf.layers.dense(concept_rep, IMAGE_EMBED_SIZE * 256, activation=tf.nn.relu)
    theta = image_dist.sample()
    theta.reshape([128, 2])
    # theta.reshape([IMAGE_EMBED_SIZE, 256])

    # Parameterize layer with theta
    x = tf.layers.dense(inputs, 512, activation=tf.nn.relu)
    x = tf.layers.dense(x, 256, activation=tf.nn.relu)
    x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    logits = dense(inputs, theta)
    # concept_rep = tf.layers.dense(concept_rep, 512, activation=tf.nn.relu)
    # theta = tf.layers.dense(concept_rep, IMAGE_EMBED_SIZE * 256, activation=tf.nn.relu)
    # theta.reshape([IMAGE_EMBED_SIZE, 256])

    # # Parameterize layer with theta
    # x = dense(inputs, theta)
    # logits = tf.layers.dense(x, 2, tf.nn.relu)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    predictions = tf.argmax(logits, axis=1)
    with tf.name_scope('metrics'):
        epoch_loss, epoch_loss_op = tf.metrics.mean(loss)
        acc, acc_op = tf.metrics.accuracy(labels, predictions)
    return logits, loss, acc, acc_op, epoch_loss, epoch_loss_op


"""
Add parameter generator, concept representation constructed using only images.
Parameter generator present.
Conjunction will be modeled post-hoc.
"""


def baseline_model_3(desc, lens, examples, inputs, labels):
    example_code = examples_encoder(examples)

    concept_rep = example_code

    # Conjunction module
    concept_rep = tf.layers.dense(concept_rep, 512, activation=tf.nn.relu)
    concept_rep = tf.layers.dense(
        concept_rep, 128 * 2 * 2, activation=tf.nn.relu)
    dist = tfd.MultivariateNormalDiag(concept_rep[:, :128 * 2],
                                      concept_rep[:, 128 * 2:])
    # Conjunction module
    # concept_rep = tf.layers.dense(concept_rep, 512, activation=tf.nn.relu)
    # theta = tf.layers.dense(concept_rep, IMAGE_EMBED_SIZE * 256, activation=tf.nn.relu)
    theta = image_dist.sample()
    theta.reshape([128, 2])
    # theta.reshape([IMAGE_EMBED_SIZE, 256])

    # Parameterize layer with theta
    x = tf.layers.dense(inputs, 512, activation=tf.nn.relu)
    x = tf.layers.dense(x, 256, activation=tf.nn.relu)
    x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    logits = dense(inputs, theta)
    # concept_rep = tf.layers.dense(concept_rep, 512, activation=tf.nn.relu)

    # theta = tf.layers.dense(concept_rep, IMAGE_EMBED_SIZE * 256, activation=tf.nn.relu)
    # theta.reshape([IMAGE_EMBED_SIZE, 256])

    # # Parameterize layer with theta
    # x = dense(inputs, theta)
    # logits = tf.layers.dense(x, 2, tf.nn.relu)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    predictions = tf.argmax(logits, axis=1)
    with tf.name_scope('metrics'):
        epoch_loss, epoch_loss_op = tf.metrics.mean(loss)
        acc, acc_op = tf.metrics.accuracy(labels, predictions)
    return logits, loss, acc, acc_op, epoch_loss, epoch_loss_op


"""
Add parameter generator, concept representation constructed using only text.
Explicitly modeled conjunction module
"""


def model_1(desc, lens, examples, inputs, labels):
    desc_0 = desc[:, 0]
    desc_1 = desc[:, 1]
    lens_0 = lens[:, 0]
    lens_1 = lens[:, 1]

    embeddings_0 = elmo(desc_0)
    embeddings_1 = elmo(desc_1)
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(128)
    initial_state = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    outputs_0, state_0 = tf.nn.dynamic_rnn(
        rnn_cell,
        embeddings_0,
        sequence_length=lens_0,
        initial_state=initial_state,
        dtype=tf.float32)

    outputs_1, state_1 = tf.nn.dynamic_rnn(
        rnn_cell,
        embeddings_1,
        sequence_length=lens_1,
        initial_state=initial_state,
        dtype=tf.float32)

    concept_rep = tf.concat([outputs0[:, -1, :], outputs1[:, -1, :]], 0)
    concept_rep = tf.layers.dense(concept_rep, 512, activation=tf.nn.relu)
    concept_rep = tf.layers.dense(
        concept_rep, 128 * 2 * 2, activation=tf.nn.relu)
    dist = tfd.MultivariateNormalDiag(concept_rep[:, :128 * 2],
                                      concept_rep[:, 128 * 2:])
    # Conjunction module
    # concept_rep = tf.layers.dense(concept_rep, 512, activation=tf.nn.relu)
    # theta = tf.layers.dense(concept_rep, IMAGE_EMBED_SIZE * 256, activation=tf.nn.relu)
    theta = image_dist.sample()
    theta.reshape([128, 2])
    # theta.reshape([IMAGE_EMBED_SIZE, 256])

    # Parameterize layer with theta
    x = tf.layers.dense(inputs, 512, activation=tf.nn.relu)
    x = tf.layers.dense(x, 256, activation=tf.nn.relu)
    x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    logits = dense(inputs, theta)
    # x = dense(inputs, theta)
    # logits = tf.layers.dense(x, 2, tf.nn.relu)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    predictions = tf.argmax(logits, axis=1)
    with tf.name_scope('metrics'):
        epoch_loss, epoch_loss_op = tf.metrics.mean(loss)
        acc, acc_op = tf.metrics.accuracy(labels, predictions)
    return logits, loss, acc, acc_op, epoch_loss, epoch_loss_op


"""
Add parameter generator, concept representation constructed using only text.
Explicitly modeled conjunction module
"""


def model_2(desc, lens, examples, inputs, labels):
    ALPHA = 1.0
    BETA = 1.0
    embeddings_0 = elmo(desc_0)
    embeddings_1 = elmo(desc_1)
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(128)
    initial_state = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    outputs_0, state_0 = tf.nn.dynamic_rnn(
        rnn_cell,
        embeddings_0,
        sequence_length=lens_0,
        initial_state=initial_state,
        dtype=tf.float32)

    outputs_1, state_1 = tf.nn.dynamic_rnn(
        rnn_cell,
        embeddings_1,
        sequence_length=lens_1,
        initial_state=initial_state,
        dtype=tf.float32)

    text_concept_rep = tf.concat([outputs0[:, -1, :], outputs1[:, -1, :]], 0)
    text_concept_rep = tf.layers.dense(
        text_concept_rep, 512, activation=tf.nn.relu)
    text_concept_rep = tf.layers.dense(
        text_concept_rep, 128 * 2 * 2, activation=tf.nn.relu)
    text_dist = tfd.MultivariateNormalDiag(text_concept_rep[:, :128 * 2],
                                           text_concept_rep[:, 128 * 2:])

    example_code = examples_encoder(examples)
    image_concept_rep = tf.layers.dense(
        example_code, 512, activation=tf.nn.relu)
    image_concept_rep = tf.layers.dense(
        example_code, 128 * 2 * 2, activation=tf.nn.relu)
    image_dist = tfd.MultivariateNormalDiag(image_concept_rep[:, :128 * 2],
                                            image_concept_rep[:, 128 * 2:])

    theta = image_dist.sample()

    theta.reshape([128, 2])

    # Parameterize layer with theta
    x = tf.layers.dense(inputs, 512, activation=tf.nn.relu)
    x = tf.layers.dense(x, 256, activation=tf.nn.relu)
    x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    logits = dense(inputs, theta)

    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    kl_loss = tfd.kl_divergence(text_dist, image_dist)
    loss = ALPHA * ce_loss + BETA * kl_loss
    predictions = tf.argmax(logits, axis=1)
    with tf.name_scope('metrics'):
        epoch_loss, epoch_loss_op = tf.metrics.mean(loss)
        acc, acc_op = tf.metrics.accuracy(labels, predictions)
    return logits, loss, acc, acc_op, epoch_loss, epoch_loss_op
