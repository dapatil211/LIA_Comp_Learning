import tensorflow as tf
import tensorflow_hub as hub
from models import baseline_model_1, baseline_model_2, baseline_model_3, model_1, model_2
from data_loader import load_all_data
import argparse


def train(model_name):
    BATCH_SIZE = 32
    EPOCHS = 1000
    SAVE_EVERY = 5
    LOG_EVERY = 10

    desc_shape = [None] if 'bl' in model_name else [None, 2]
    desc_placeholder = tf.placeholder(tf.string, desc_shape)
    len_placeholder = tf.placeholder(tf.int32, desc_shape)
    examples_placeholder = tf.placeholder(tf.float32, [None, 4, 1280])
    inputs_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 3])
    labels_placeholder = tf.placeholder(tf.int32, [None])

    dataset = tf.data.Dataset.from_tensor_slices(
        (desc_placeholder, len_placeholder, examples_placeholder,
         inputs_placeholder, labels_placeholder))
    eval_dataset = dataset.batch(BATCH_SIZE)
    eval_dataset = eval_dataset.prefetch(BATCH_SIZE)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset.prefetch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()
    eval_iterator = eval_dataset.make_initializable_iterator()

    if model_name == 'bl1':
        model = baseline_model_1
    elif model_name == 'bl2':
        model = baseline_model_2
    elif model_name == 'bl3':
        model = baseline_model_3
    elif model_name == 'm1':
        model = model_1
    elif model_name == 'm2':
        model = model_2

    train_logits, train_loss, train_accuracy, train_acc_op, train_epoch_loss, train_epoch_loss_op = model(
        *iterator.get_next())
    with tf.variable_scope('', reuse=True):
        eval_logits, eval_loss, eval_accuracy, eval_acc_op, eval_epoch_loss, eval_epoch_loss_op = model(
            *eval_iterator.get_next(), train=False)
    reset_metrics = tf.variables_initializer(
        [v for v in tf.local_variables() if 'metrics' in v.name])
    optimize = tf.train.AdamOptimizer(0.005).minimize(train_loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    desc_train, len_train, examples_train, inputs_train, labels_train = load_all_data(
        'complearn/train')
    desc_val, len_val, examples_val, inputs_val, labels_val = load_all_data(
        'complearn/val')
    desc_test, len_test, examples_test, inputs_test, labels_test = load_all_data(
        'complearn/test')

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(EPOCHS):
            step = 0
            sess.run(
                [reset_metrics, iterator.initializer],
                feed_dict={
                    desc_placeholder: desc_train,
                    len_placeholder: len_train,
                    examples_placeholder: examples_train,
                    inputs_placeholder: inputs_train,
                    labels_placeholder: labels_train
                })
            while True:
                try:
                    sess.run([optimize, train_acc_op, train_epoch_loss_op])
                    step += 1
                    if step % LOG_EVERY == 0:
                        [tr_loss, tr_accuracy] = sess.run(
                            [train_epoch_loss, train_accuracy])
                        print('Step %d\ttrain loss:%f\ttrain accuracy:%f' %
                              (step, tr_loss, tr_accuracy))
                except tf.errors.OutOfRangeError:
                    break
            sess.run(
                [eval_iterator.initializer],
                feed_dict={
                    desc_placeholder: desc_val,
                    len_placeholder: len_val,
                    examples_placeholder: examples_val,
                    inputs_placeholder: inputs_val,
                    labels_placeholder: labels_val
                })
            while True:
                try:
                    sess.run([eval_acc_op, eval_epoch_loss_op])
                except tf.errors.OutOfRangeError:
                    break
            [acc_train, loss_train, acc_val, loss_val] = \
                sess.run([train_accuracy, train_epoch_loss, eval_accuracy, eval_epoch_loss])
            print(
                'Epoch %d\tTraining Loss: %f\tTraining Accuracy: %f\tValidation Loss: %f\tValidation Accuracy: %f'
                % (epoch, loss_train, acc_train, loss_val, acc_val))
            if epoch % SAVE_EVERY:
                save_path = saver.save(sess, "./models/model.ckpt")


def main():
    parser = argparse.ArgumentParser(description='Compositional Learning')
    parser.add_argument(
        '-m',
        '--model',
        choices=['bl1', 'bl2', 'bl3', 'm1', 'm2'],
        default='bl1')
    args = parser.parse_args()
    train(args.model)


if __name__ == '__main__':
    main()