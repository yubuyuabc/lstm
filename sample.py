# -*- coding: utf-8 -*-

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from read_utils import TextConverter
from model import CharRNN
import os

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_integer('lstm_size', 256, 'size of hidden state of lstm')
tf.compat.v1.flags.DEFINE_integer('num_layers', 3, 'number of lstm layers')
tf.compat.v1.flags.DEFINE_boolean('use_embedding',True, 'whether to use embedding')
tf.compat.v1.flags.DEFINE_integer('embedding_size', 256, 'size of embedding')
tf.compat.v1.flags.DEFINE_string('converter_path', './model/novel/converter.pkl', 'model/name/converter.pkl')
tf.compat.v1.flags.DEFINE_string('checkpoint_path', './model/novel', 'checkpoint path')
tf.compat.v1.flags.DEFINE_string('start_string', '中神通', 'use this string to start generating')
tf.compat.v1.flags.DEFINE_integer('max_length', 1000, 'max length to generate')


def main(_):
    #FLAGS.start_string = FLAGS.start_string.decode('utf-8')
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.compat.v1.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(converter.vocab_size, sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)

    start = converter.text_to_arr(FLAGS.start_string)
    arr = model.sample(FLAGS.max_length, start, converter.vocab_size)
    print(converter.arr_to_text(arr))


if __name__ == '__main__':
    tf.compat.v1.app.run()
