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
flags.DEFINE_boolean("resume", True, "Whether to resume last time's training")
flags.DEFINE_integer("sentence_size", "0", "0")
flags.DEFINE_integer("question_words", "0", "0")
FLAGS = flags.FLAGS


def main(_):
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
	train_q, valid_q, train_a, valid_a, train_p, valid_p = train_test_split(
		train_q, train_a, train_p, test_size=.11, random_state=0)

	n_training = train_q.shape[0]
	n_testing = test_q.shape[0]
	n_validation = valid_q.shape[0]

	print("Training Size", n_training)
	print("Validation Size", n_validation)
	print("Testing Size", n_testing)

	print(flags.FLAGS.__flags)

	# batch_id
	# batches = [(start, end) for start, end in batches] abandon last few examples
	batches = list(zip(range(0, n_training - FLAGS.batch_size, FLAGS.batch_size),
	                   range(FLAGS.batch_size, n_training, FLAGS.batch_size)))

	kb1_triples = multi_kb.kb1.triples
	kb2_triples = multi_kb.kb2.triples
	alignments = multi_kb.align_array

	with tf.Session() as sess:
		model = M_IRN(FLAGS, multi_kb, sess)

		if FLAGS.resume:
			model.load()

		print("knowledge base 1 size", kb1_triples.shape[0])
		print("knowledge base 2 size", kb2_triples.shape[0])
		kg1_embedding_batches = list(zip(range(0, kb1_triples.shape[0] - FLAGS.batch_size, FLAGS.batch_size),
		                                 range(FLAGS.batch_size, kb1_triples.shape[0], FLAGS.batch_size)))
		kg2_embedding_batches = list(zip(range(0, kb2_triples.shape[0] - FLAGS.batch_size, FLAGS.batch_size),
		                                 range(FLAGS.batch_size, kb2_triples.shape[0], FLAGS.batch_size)))
		alignment_batches = list(zip(range(0, alignments.shape[0] - FLAGS.batch_size, FLAGS.batch_size),
		                             range(FLAGS.batch_size, alignments.shape[0], FLAGS.batch_size)))

		pre_val_predictions = model.predict(valid_q, valid_p)
		best_val_epoch = -1
		best_val_accuracy = multi_accuracy(valid_p, pre_val_predictions, multi_kb)

		for t in range(1, FLAGS.r_epoch + 1):
			start = time.time()
			np.random.shuffle(batches)
			embedding_cost_min = float("inf")
			alignment_cost_min = float("inf")
			for i in range(FLAGS.e_epoch):
				np.random.shuffle(kg1_embedding_batches)
				np.random.shuffle(kg2_embedding_batches)
				embedding_total_cost = 0.0
				for s, e in kg1_embedding_batches:
					embedding_total_cost += model.batch_train_kg1_embedding(kb1_triples[s:e])
				for s, e in kg2_embedding_batches:
					embedding_total_cost += model.batch_train_kg2_embedding(kb2_triples[s:e])
				embedding_cost_min = min(embedding_cost_min, embedding_total_cost)
				for j in range(1, FLAGS.a_epoch + 1):
					np.random.shuffle(alignment_batches)
					alignment_total_cost = 0.0
					for s, e in alignment_batches:
						alignment_total_cost += model.batch_train_alignment(alignments[s:e])
					alignment_cost_min = min(alignment_cost_min, alignment_total_cost)

			reasoning_total_cost = 0.0
			for s, e in batches:
				reasoning_total_cost += model.batch_train_inference(train_q[s:e], train_a[s:e],
				                                                    np.argmax(train_a[s:e], axis=1), train_p[s:e])

			train_predictions = model.predict(train_q, train_p)
			train_accuracy = multi_accuracy(train_p, train_predictions, multi_kb)

			valid_predictions = model.predict(valid_q, valid_p)
			valid_accuracy = multi_accuracy(valid_p, valid_predictions, multi_kb)

			if valid_accuracy[-3] > best_val_accuracy[-3]:
				best_val_epoch = t
				best_val_accuracy = valid_accuracy
				model.store()

			print('-----------------------')
			print('Epoch', t)
			print('timing', (time.time() - start))
			print('Embedding total cost:', embedding_cost_min)
			print('Alignment total cost:', alignment_cost_min)
			print('Reasoning total cost:', reasoning_total_cost)
			print('Training Accuracy:', train_accuracy[-3])
			print('Training Alignment Accuracy:', train_accuracy[-1])
			print('Training Alignment Missing rate:', train_accuracy[-2])
			print('Training Accuracy for whole Path:', train_accuracy[0:-2])
			print('Validation Accuracy:', valid_accuracy[-3])
			print('Validation Alignment Accuracy:', valid_accuracy[-1])
			print('Validation Alignment Missing rate:', valid_accuracy[-2])
			print('Validation Accuracy for whole Path:', valid_accuracy[0:-2])
			print('Best Validation epoch , accuracy for whole path, alignment accuracy and missing rate:',
			      best_val_epoch, best_val_accuracy[0:-2], best_val_accuracy[-1], best_val_accuracy[-2])
			print('-----------------------')


if __name__ == '__main__':
	tf.app.run(main)
