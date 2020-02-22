# pylint: disable-all
# coding: utf-8
import os
import tensorflow as tf
import numpy as np
import time
from sklearn.model_selection import train_test_split
from utils import multi_accuracy, create_batch
from model import M_IRN
from data_utils import process_dataset, MultiKnowledgeBase, KnowledgeBase

flags = tf.flags

flags.DEFINE_integer("embedding_dimension", 64, "KG vector dimension [64]")
flags.DEFINE_integer("batch_size", 50, "batch size to use during training [50]")
flags.DEFINE_integer("r_epoch", 2000, "number of epochs to use during training [2000]")
flags.DEFINE_integer("e_epoch", 10, "number of middle epochs for embedding training [10]")
flags.DEFINE_integer("a_fold", 5, "number of inner epochs for alignment training [3]")
flags.DEFINE_float("max_grad_norm", 20, "clip gradients to this norm [10]")
flags.DEFINE_float("alignment_ratio", 1, "Alignment seeds ratio [0.5]")
flags.DEFINE_float("lr", 0.001, "Learning rate [0.001]")
flags.DEFINE_float("epsilon", 1e-8, "Epsilon for Adam Optimizer [1e-8]")
flags.DEFINE_string("dataset", "EN_en_zh_en", "dataset name")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "checkpoint directory")
flags.DEFINE_string("data_dir", "data", "dataset directory")
flags.DEFINE_boolean("resume", False, "Whether to resume last time's training")
flags.DEFINE_boolean("direct_align", True, "Replace entity embedding alignment with actual alignment")
flags.DEFINE_boolean("dual_matrices", False, "Whether to use two transfer matrices")
flags.DEFINE_boolean("pre_kg", False, "Whether to train KG and alignment additionally before reasoning")
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
	is_flip =  True if dir in ["zh_fr", "en_fr", "zh_en"] else False

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
	train_q, valid_q, train_p, valid_p = train_test_split(train_q, train_p, test_size=.11, random_state=0)

	n_training = train_q.shape[0]
	n_testing = test_q.shape[0]
	n_validation = valid_q.shape[0]

	print("Training Size", n_training)
	print("Validation Size", n_validation)
	print("Testing Size", n_testing)

	# batch_id
	# batches = [(start, end) for start, end in batches] abandon last few examples
	tr_batches = create_batch(n_training, FLAGS.batch_size)
	v_batches =  create_batch(n_validation, FLAGS.batch_size)
	t_batches =  create_batch(n_testing, FLAGS.batch_size)

	kb1_triples = multi_kb.kb1.triples
	kb2_triples = multi_kb.kb2.triples
	a_seeds = multi_kb.a_array_train
	a_tests = multi_kb.a_array_test

	with tf.Session() as sess:
		model = M_IRN(FLAGS, multi_kb, sess)

		if FLAGS.resume:
			model.load()

		print("knowledge base 1 size", kb1_triples.shape[0])
		print("knowledge base 2 size", kb2_triples.shape[0])
		kg1_embedding_batches =  create_batch(kb1_triples.shape[0], FLAGS.batch_size)
		kg2_embedding_batches = create_batch(kb2_triples.shape[0], FLAGS.batch_size)
		align_tr_batches = create_batch(a_seeds.shape[0], FLAGS.batch_size)
		align_t_batches = create_batch(a_tests.shape[0], FLAGS.batch_size)

		if FLAGS.pre_kg:
			kg1_embedding_cost = 0.0
			kg2_embedding_cost = 0.0
			alignment_cost = 0.0
			best_1_2 = 0.0
			best_2_1 = 0.0
			best_1_2_epoch = best_2_1_epoch = 0
			for i in range(1, 400 + 1):
				start = time.time()
				print("MIRN KG epoch {} training...".format(i))
				np.random.shuffle(kg1_embedding_batches)
				np.random.shuffle(kg2_embedding_batches)
				kg1_embedding_total_cost = 0.0
				kg2_embedding_total_cost = 0.0
				for s, e in kg1_embedding_batches:
					kg1_embedding_total_cost += model.batch_train_kg1_embedding(kb1_triples[s:e])
				kg1_embedding_cost = kg1_embedding_total_cost
				for s, e in kg2_embedding_batches:
					kg2_embedding_total_cost += model.batch_train_kg2_embedding(kb2_triples[s:e])
				kg2_embedding_cost = kg2_embedding_total_cost
				for j in range(FLAGS.a_fold):
					np.random.shuffle(align_tr_batches)
					alignment_total_cost = 0.0
					for s, e in align_tr_batches:
						alignment_total_cost += model.batch_train_alignment(a_seeds[s:e])
					alignment_cost = alignment_total_cost
				align_accu_1_2, align_accu_2_1 = model.align_res(a_tests, align_t_batches)

				if align_accu_1_2 > best_1_2:
					best_1_2_epoch = i
					best_1_2 = align_accu_1_2

				if align_accu_2_1 > best_2_1:
					best_2_1_epoch = i
					best_2_1 = align_accu_2_1

				print('--------------------------------------------------------------------------------------------')
				print('Epoch', i)
				print('Timing', (time.time() - start))
				print('Embedding total cost for KG1:', kg1_embedding_cost)
				print('Embedding total cost for KG2:', kg2_embedding_cost)
				print('Alignment total cost:', alignment_cost)
				print('Alignment test accuracy: 1-2: {}, 2-1: {}.'.format(align_accu_1_2, align_accu_2_1))
				print('Best 1-2 alignment epoch & accuracy: ', best_1_2_epoch, best_1_2)
				print('Best 2-1 alignment epoch & accuracy: ', best_2_1_epoch, best_2_1)
				print('--------------------------------------------------------------------------------------------')

		pre_v_preds = model.predict(valid_q, valid_p, v_batches)
		pre_t_preds = model.predict(test_q, test_p, t_batches)
		best_v_ep = -1
		best_v_accu, best_v_al = multi_accuracy(valid_p, pre_v_preds, multi_kb, FLAGS.steps, FLAGS.hops, FLAGS.lan_labels)
		best_t_accu, best_t_al = multi_accuracy(test_p, pre_t_preds, multi_kb, FLAGS.steps, FLAGS.hops, FLAGS.lan_labels)

		for t in range(1, FLAGS.r_epoch + 1):
			start = time.time()
			np.random.shuffle(tr_batches)

			kg1_embedding_cost = kg2_embedding_cost = alignment_cost = 0.0

			print("MIRN multi epoch {} training...".format(t))

			# e_epoch = 100 if t == 1 else FLAGS.e_epoch

			for i in range(1, FLAGS.e_epoch + 1):
				np.random.shuffle(kg1_embedding_batches)
				np.random.shuffle(kg2_embedding_batches)
				kg1_embedding_total_cost = 0.0
				kg2_embedding_total_cost = 0.0
				for s, e in kg1_embedding_batches:
					kg1_embedding_total_cost += model.batch_train_kg1_embedding(kb1_triples[s:e])
				kg1_embedding_cost = kg1_embedding_total_cost
				for s, e in kg2_embedding_batches:
					kg2_embedding_total_cost += model.batch_train_kg2_embedding(kb2_triples[s:e])
				kg2_embedding_cost = kg2_embedding_total_cost
			for j in range(FLAGS.a_fold):
				np.random.shuffle(align_tr_batches)
				alignment_total_cost = 0.0
				for s, e in align_tr_batches:
					alignment_total_cost += model.batch_train_alignment(a_seeds[s:e])
				alignment_cost = alignment_total_cost

			reasoning_total_cost = 0.0
			for s, e in tr_batches:
				reasoning_total_cost += model.batch_train_inference(train_q[s:e], train_p[s:e])

			tr_preds = model.predict(train_q, train_p, tr_batches)
			tr_accu, tr_al = multi_accuracy(train_p, tr_preds, multi_kb, FLAGS.steps, FLAGS.hops, FLAGS.lan_labels)
			v_preds = model.predict(valid_q, valid_p, v_batches)
			v_accu, v_al = multi_accuracy(valid_p, v_preds, multi_kb, FLAGS.steps, FLAGS.hops, FLAGS.lan_labels)
			t_preds = model.predict(test_q, test_p, t_batches)
			t_accu, t_al = multi_accuracy(test_p, t_preds, multi_kb, FLAGS.steps, FLAGS.hops, FLAGS.lan_labels)
			align_accu_1_2, align_accu_2_1 = model.align_res(a_tests, align_t_batches)

			if v_accu[-1] > best_v_accu[-1]:
				best_v_ep = t
				best_v_accu = v_accu
				best_v_al = v_al
				model.store()
				best_t_accu = t_accu
				best_t_al = t_al

			print('--------------------------------------------------------------------------------------------'
			      '--------------------------------------------------------------------------------------------')
			print('Epoch', t)
			print('Timing', (time.time() - start))
			print('Embedding total cost for KG1:', kg1_embedding_cost)
			print('Embedding total cost for KG2:', kg2_embedding_cost)
			print('Alignment total cost:', alignment_cost)
			print('Alignment test accuracy: 1-2: {}, 2-1: {}.'.format(align_accu_1_2, align_accu_2_1))
			print('Reasoning total cost:', reasoning_total_cost)
			print('Training Accuracy:', tr_accu, tr_al)
			print('Validation Accuracy:', v_accu, v_al)
			print('Test Accuracy:', t_accu, t_al)
			print('Best Validation epoch & accuracy for whole path:', best_v_ep, best_v_accu, best_v_al)
			print('Test accuracy for whole path under best Validation epoch:', best_t_accu, best_t_al)
			print('--------------------------------------------------------------------------------------------'
			      '--------------------------------------------------------------------------------------------')


if __name__ == '__main__':
	tf.app.run(main)
