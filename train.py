import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from read_utils import TextConverter, batch_generator
from model import CharRNN
import os
import codecs

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string('name', 'novel', 'name of the model')
tf.compat.v1.flags.DEFINE_integer('num_seqs', 32, 'number of seqs in one batch')
tf.compat.v1.flags.DEFINE_integer('num_steps', 80, 'length of one seq')
tf.compat.v1.flags.DEFINE_integer('lstm_size', 256, 'size of hidden state of lstm')
tf.compat.v1.flags.DEFINE_integer('num_layers', 3, 'number of lstm layers')
tf.compat.v1.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.compat.v1.flags.DEFINE_integer('embedding_size', 256, 'size of embedding')
tf.compat.v1.flags.DEFINE_float('learning_rate', 0.005, 'learning_rate')
tf.compat.v1.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.compat.v1.flags.DEFINE_string('input_file', './data/novel.txt', 'utf8 encoded text file')
tf.compat.v1.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
tf.compat.v1.flags.DEFINE_integer('save_every_n', 500, 'save the model every n steps')
tf.compat.v1.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
tf.compat.v1.flags.DEFINE_integer('max_vocab', 6000, 'max char number')


def main(_):
    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text, FLAGS.max_vocab)
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)
    print(converter.vocab_size)
    model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.compat.v1.app.run()
