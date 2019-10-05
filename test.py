# pylint: disable-all
# coding: utf-8
import os
import tensorflow as tf
import numpy as np
import time

from sklearn.model_selection import train_test_split
from utils import multi_accuracy
from model import M_IRN
from data_utils import process_dataset, MultiKnowledgeBase, KnowledgeBase

flags = tf.flags

flags.DEFINE_integer("embedding_dimension", 64, "words vector dimension [50]")
flags.DEFINE_integer("batch_size", 50, "batch size to use during training [50]")
flags.DEFINE_integer("r_epoch", 5000, "number of epochs to use during training [5000]")
flags.DEFINE_integer("e_epoch", 3, "number of middle epochs for embedding training [5]")
flags.DEFINE_integer("a_epoch", 2, "number of inner epochs for alignment training [3]")
flags.DEFINE_float("max_grad_norm", 20, "clip gradients to this norm [20]")
flags.DEFINE_string("dataset", "en_en_zh", "question language + batch_kb1 language + kb2 language, "
                                           "options are en_en_zh/en_zh_en/zh_en_zh/zh_zh_en")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "checkpoint directory")
flags.DEFINE_string("data_dir", "data", "dataset directory")
flags.DEFINE_integer("sentence_size", "0", "0")
flags.DEFINE_integer("question_words", "0", "0")
FLAGS = flags.FLAGS

def test():
    FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.dataset)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    dataset_substrings = FLAGS.dataset.split("_")
    kb1_file = '%s/%s_triples.txt' % (FLAGS.data_dir, dataset_substrings[1])
    kb2_file = '%s/%s_triples.txt' % (FLAGS.data_dir, dataset_substrings[2])
    alignment_file = '%s/%s_%s_alignment.txt' % (FLAGS.data_dir, dataset_substrings[1], dataset_substrings[2])
    data_file = '%s/%s.txt' % (FLAGS.data_dir, FLAGS.dataset)

    start = time.time()
    # build and store knowledge bases
    kb1 = KnowledgeBase(kb1_file, name="kb1")
    kb2 = KnowledgeBase(kb2_file, name="kb2")
    multi_kb = MultiKnowledgeBase(kb1, kb2, alignment_file)
    questions_ids, answers_id_one_hot, paths_ids, questions_strings, answers_ids, path_strings, FLAGS.sentence_size, \
    word2id, FLAGS.question_words = process_dataset(data_file, multi_kb, is_cn=(dataset_substrings[0] == "zh"))
    print("read data cost %f seconds" % (time.time() - start))

    train_q, test_q, train_a, test_a, train_p, test_p = train_test_split(
        questions_ids, answers_id_one_hot, paths_ids, test_size=.1, random_state=123)

    n_training = train_q.shape[0]
    n_testing = test_q.shape[0]

    print("Training Size", n_training)
    print("Testing Size", n_testing)

    print(flags.FLAGS)

    with tf.Session() as sess:
        model = M_IRN(FLAGS, multi_kb, sess)
        model.load()
        test_predictions = model.predict(test_q, test_p)
        test_acc= multi_accuracy(test_p, test_predictions, multi_kb)

        print('-----------------------')
        print('Test Data', data_file)
        print('Test Accuracy:', test_acc[-3])
        print('Alignment Accuracy:', test_acc[-1])
        print('Alignment Missing rate:', test_acc[-2])
        print('Test Accuracy for whole Path:', test_acc[0:-2])
        print('-----------------------')

if __name__ == '__main__':
    test()