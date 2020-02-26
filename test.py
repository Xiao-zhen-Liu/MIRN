# pylint: disable-all
# coding: utf-8
import os
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from utils import multi_accuracy, create_batch
from model import M_IRN
from data_utils import process_dataset, MultiKnowledgeBase, KnowledgeBase, recover_predictions

flags = tf.flags

flags.DEFINE_integer("embedding_dimension", 64, "KG vector dimension [50]")
flags.DEFINE_integer("batch_size", 50, "batch size to use during training [50]")
flags.DEFINE_float("max_grad_norm", 10, "clip gradients to this norm [10]")
flags.DEFINE_float("alignment_ratio", 1, "Alignment seeds ratio [0.5]")
flags.DEFINE_float("lr", 0.001, "Learning rate [0.001]")
flags.DEFINE_float("ar", 2.5, "Learning rate multiplication for AM [2.5]")
flags.DEFINE_float("epsilon", 1e-8, "Epsilon for Adam Optimizer [1e-8]")
flags.DEFINE_string("dataset", "EN_en_zh_en", "dataset name")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "checkpoint directory")
flags.DEFINE_string("data_dir", "data", "dataset directory")
flags.DEFINE_boolean("direct_align", False, "Replace entity embedding alignment with actual alignment")
flags.DEFINE_boolean("dual_matrices", False, "Whether to use two transfer matrices")
flags.DEFINE_integer("sentence_size", 0, "")
flags.DEFINE_integer("question_words", 0, "")
flags.DEFINE_integer("hops", 0, "")
flags.DEFINE_string("lan_que", "", "")
flags.DEFINE_list("lan_labels", [], "")
flags.DEFINE_list("steps", [], "")

FLAGS = flags.FLAGS


def main(_):
    FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.dataset)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    d_labels = FLAGS.dataset.split("_")
    additional = False if d_labels[-1] in {"zh", "en", "fr"} else True
    lan_1 = d_labels[1]
    kb1_file = '{}/{}_{}_KB.txt'.format(FLAGS.data_dir, lan_1, d_labels[-1]) if additional else '{}/{}_KB.txt'. \
        format(FLAGS.data_dir, lan_1)
    lan_2 = ""
    for label in d_labels[1:]:
        if label != lan_1:
            lan_2 = label
            break
    kb2_file = '{}/{}_{}_KB.txt'.format(FLAGS.data_dir, lan_2, d_labels[-1]) if additional else '{}/{}_KB.txt'. \
        format(FLAGS.data_dir, lan_2)
    data_file = '{}/{}.txt'.format(FLAGS.data_dir, FLAGS.dataset)

    dir = lan_1 + "_" + lan_2
    is_flip = True if dir in ["zh_fr", "en_fr", "zh_en"] else False

    align_train_file = '{}/{}_{}_train.txt'.format(FLAGS.data_dir, lan_2 if is_flip else lan_1,
                                                   lan_1 if is_flip else lan_2)
    align_test_file = '{}/{}_{}_test.txt'.format(FLAGS.data_dir, lan_2 if is_flip else lan_1,
                                                 lan_1 if is_flip else lan_2)  # Also used for table lookup

    start = time.time()
    print("Loading data...")

    # build and store knowledge bases
    kb1 = KnowledgeBase(kb1_file, name="kb1")
    kb2 = KnowledgeBase(kb2_file, name="kb2")
    multi_kb = MultiKnowledgeBase(kb1, kb2, align_train_file, align_test_file, is_flip)

    q_ids, p_ids, q_strs, p_strs, qw2id, FLAGS.sentence_size, FLAGS.question_words, FLAGS.hops, \
    FLAGS.lan_que, FLAGS.lan_labels, FLAGS.steps = process_dataset(data_file, multi_kb)

    print("Data loading cost {} seconds".format(time.time() - start))

    train_q, test_q, train_p, test_p = train_test_split(q_ids, p_ids, test_size=.1, random_state=123)

    n_testing = test_q.shape[0]

    print("Testing Size", n_testing)

    t_batches = create_batch(n_testing, FLAGS.batch_size)
    a_tests = multi_kb.a_array_test
    align_t_batches = create_batch(a_tests.shape[0], FLAGS.batch_size)

    with tf.Session() as sess:
        model = M_IRN(FLAGS, multi_kb, sess)

        model.load()
        t_preds = model.predict(test_q, test_p, t_batches)
        t_accu, t_al = multi_accuracy(test_p, t_preds, multi_kb, FLAGS.steps, FLAGS.hops, FLAGS.lan_labels)
        align_accu_1_2, align_accu_2_1 = model.align_res(a_tests)
        recov_name = data_file.strip(".txt") + "_predictions.txt"

        print('-----------------------')
        print('Test Data', data_file)
        print('Test Accuracy:', t_accu, t_al)
        print('Alignment test accuracy: 1-2: {}, 2-1: {}.'.format(align_accu_1_2, align_accu_2_1))
        print('-----------------------')
        print("prediction results saved in {}.".format(recov_name))
        recover_predictions(recov_name, t_preds, multi_kb, FLAGS.hops, FLAGS.lan_labels,
                            FLAGS.steps)

if __name__ == '__main__':
    tf.app.run(main)
