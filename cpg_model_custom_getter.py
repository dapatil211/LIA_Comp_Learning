import tensorflow as tf
from cpg_initializer import CPGInitializer
from data_loader import load_dil
from comp_module import ContextParser
initializer = CPGInitializer()
tf.enable_eager_execution()
tfe = tf.contrib.eager


def model_fn(images, labels, contexts):
    # print(contexts)
    with tf.variable_scope(
            'model',
            custom_getter=lambda getter, name, shape, *args, **kwargs:
            initializer.getter(contexts, getter, name, shape, *args, **kwargs)):
        x = tf.layers.conv2d(images, 64, 3, activation=tf.nn.relu, name='conv1')
        x = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu, name='conv2')
        x = tf.reduce_mean(x, [1, 2])
        x = tf.layers.dense(x, 32, activation=tf.nn.relu, name='dense1')
        logits = tf.layers.dense(x, 2, activation=tf.nn.relu, name='dense2')

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    return predictions, loss


def input_fn(folder):
    inputs, labels, descs = load_dil(folder)
    num_examples = inputs.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, descs))
    dataset = dataset.cache()
    dataset = dataset.shuffle(num_examples * .7)
    dataset = dataset.batch(1)
    # dataset = dataset.map(parse_context)
    return dataset, num_examples


def train():
    EPOCHS = 2
    optimizer = tf.train.AdamOptimizer(0.001)
    global_container = tfe.EagerVariableStore()
    with global_container.as_default():
        parser = ContextParser()

    for epoch in range(EPOCHS):
        train_dataset, num_train = input_fn('complearn/train')
        val_dataset, num_val = input_fn('complearn/val')
        tc_train = 0.0
        tc_val = 0.0
        loss_train = 0.0
        loss_val = 0.0
        print('Start epoch %d' % epoch)
        for image, label, desc in train_dataset:
            current_container = tfe.EagerVariableStore()

            with tf.GradientTape() as tape:
                with current_container.as_default():
                    context = parser.parse_descs(desc)
                    predictions, loss = model_fn(image, label, context)
                tc_train += tf.reduce_sum(
                    tf.cast(tf.equal(predictions, label), tf.float32))
                loss_train += tf.reduce_sum(loss)
            var_list = global_container.variables() + [
                var[1] for var in initializer.generation_params.values()
            ]  #+ current_container.variables()
            grads = tape.gradient(loss, var_list)
            print(grads)
            print(len(var_list))
            optimizer.apply_gradients(
                zip(grads, var_list),
                global_step=tf.train.get_or_create_global_step())
        for image, label, desc in train_dataset:
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