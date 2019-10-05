# coding=utf-8
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from data_utils import MultiKnowledgeBase
from utils import add_gradient_noise, zero_nil_slot


class M_IRN(object):
	def __init__(self, config, multi_kb: MultiKnowledgeBase, sess):
		self._margin = 4
		self._dataset = config.dataset
		self._batch_size = config.batch_size
		self._sentence_size = config.sentence_size
		self._embedding_size = config.embedding_dimension
		self._max_grad_norm = config.max_grad_norm
		self._multi_kb = multi_kb
		self._vocab_size = config.question_words
		self._rel_size_1 = self._multi_kb.kb1.n_relations
		self._rel_size_2 = self._multi_kb.kb2.n_relations
		self._ent_size_1 = self._multi_kb.kb1.n_entities
		self._ent_size_2 = self._multi_kb.kb2.n_entities
		self._init = tf.contrib.layers.xavier_initializer()
		self._opt = tf.train.AdamOptimizer()
		self._name = "M_IRN"
		self._checkpoint_dir = config.checkpoint_dir + '/' + self._name

		if not os.path.exists(self._checkpoint_dir):
			os.makedirs(self._checkpoint_dir)

		self._build_inputs()
		self._build_vars()
		self._saver = tf.train.Saver(max_to_keep=1)

		# kg1 train and loss
		kg1_batch_loss = self._kg1_to_train()
		kg1_loss_op = tf.reduce_sum(kg1_batch_loss, name="KG1_loss_op")
		kg1_grads_and_vars = self._opt.compute_gradients(kg1_loss_op,
		                                                 [self._kg1_entity_embedding, self._kg1_relation_embedding,
		                                                  self._kg1_Mse])
		kg1_nil_grads_and_vars = []
		for g, v in kg1_grads_and_vars:
			if v.name in self._nil_vars:  # not _kg1_Mse
				kg1_nil_grads_and_vars.append((zero_nil_slot(g), v))
			else:
				kg1_nil_grads_and_vars.append((g, v))
		print("KG1_grads_and_vars")
		for g, v in kg1_nil_grads_and_vars:
			print(g, v.name)
		kg1_train_op = self._opt.apply_gradients(kg1_grads_and_vars, name="kg1_train_op")

		# kg2 train and loss
		kg2_batch_loss = self._kg2_to_train()
		kg2_loss_op = tf.reduce_sum(kg2_batch_loss, name="kg2_loss_op")
		kg2_grads_and_vars = self._opt.compute_gradients(kg2_loss_op,
		                                                 [self._kg2_entity_embedding, self._kg2_relation_embedding,
		                                                  self._kg2_Mse])
		kg2_nil_grads_and_vars = []
		for g, v in kg2_grads_and_vars:
			if v.name in self._nil_vars:  # not _kg2_Mse
				kg2_nil_grads_and_vars.append((zero_nil_slot(g), v))
			else:
				kg2_nil_grads_and_vars.append((g, v))
		print("kg2_grads_and_vars")
		for g, v in kg2_nil_grads_and_vars:
			print(g, v.name)
		kg2_train_op = self._opt.apply_gradients(kg2_grads_and_vars, name="kg2_train_op")

		# alignment train and loss
		alignment_batch_loss = self._multi_kg_to_train()
		alignment_loss_op = tf.reduce_sum(alignment_batch_loss, name="alignment_loss_op")
		alignment_train_op = self._opt.minimize(alignment_loss_op)

		# cross entropy as loss for inference:
		batch_loss, inference_path = self._inference()  # (batch_size, 1), (batch_size, 6)
		inference_loss_op = tf.reduce_sum(batch_loss, name="inference_loss_op")
		inference_params = [self._question_embedding, self._kg1_Mrq, self._kg1_Mrs, self._kg2_Mrq, self._kg2_Mrs,
		                    self._kg1_Mqe, self._kg2_Meq]
		inference_grads_and_vars = self._opt.compute_gradients(inference_loss_op, inference_params)
		inference_grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in inference_grads_and_vars if
		                            g is not None]
		inference_grads_and_vars = [(add_gradient_noise(g), v) for g, v in inference_grads_and_vars]
		inference_nil_grads_and_vars = []
		for g, v in inference_grads_and_vars:
			if v.name in self._nil_vars:
				inference_nil_grads_and_vars.append((zero_nil_slot(g), v))
			else:
				inference_nil_grads_and_vars.append((g, v))
		print("inference_grads_and_vars")
		for g, v in inference_nil_grads_and_vars:
			print(g, v.name)
		inference_train_op = self._opt.apply_gradients(inference_nil_grads_and_vars, name="inference_train_op")

		# predict ops
		inference_predict_op = inference_path

		# assign ops
		self.kg1_loss_op = kg1_loss_op
		self.kg1_train_op = kg1_train_op
		self.kg2_loss_op = kg2_loss_op
		self.kg2_train_op = kg2_train_op
		self.alignment_loss_op = alignment_loss_op
		self.alignment_train_op = alignment_train_op
		self.inference_loss_op = inference_loss_op
		self.inference_predict_op = inference_predict_op
		self.inference_train_op = inference_train_op

		init_op = tf.global_variables_initializer()
		self._sess = sess
		self._sess.run(init_op)

	def _build_inputs(self):
		self._kbs_1 = tf.placeholder(tf.int32, [None, 3], name="KBs_1")
		self._kbs_2 = tf.placeholder(tf.int32, [None, 3], name="KBs_2")
		self._alignment_seeds = tf.placeholder(tf.int32, [None, 2], name="alignment_seeds")
		self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
		# IDs for e1(l1), r1(l1), e2(l1), e2(l2), r2(l2), e3(l2, answer)
		self._paths = tf.placeholder(tf.int32, [None, 6], name="paths")
		self._answers = tf.placeholder(tf.int32, [None, self._ent_size_2], name="answers")  # ID-hot for answer
		self._answers_id = tf.placeholder(tf.int32, [None], name="answers_id")  # id for answer
		self._padding_1 = tf.placeholder(tf.int32, [None], name="padding_1")  # for id_padding
		self._padding_2 = tf.placeholder(tf.int32, [None], name="padding_2")
		self._zeros = tf.placeholder(tf.float32, [None], name="zeros")
		self._isTrain = tf.placeholder(tf.int32, name="ground_truth")

	def _build_vars(self):
		with tf.variable_scope(self._name):
			nil_word_slot = tf.zeros([1, self._embedding_size])
			nil_rel_slot = tf.zeros([1, self._embedding_size])
			kg1_entity_slot = tf.concat(axis=0, values=[nil_word_slot, self._init([self._ent_size_1 - 1,
			                                                                       self._embedding_size])])
			kg2_entity_slot = tf.concat(axis=0, values=[nil_word_slot, self._init([self._ent_size_2 - 1,
			                                                                       self._embedding_size])])
			kg1_relation_slot = tf.concat(axis=0, values=[nil_rel_slot, self._init([self._rel_size_1 - 1,
			                                                                        self._embedding_size])])
			kg2_relation_slot = tf.concat(axis=0, values=[nil_rel_slot, self._init([self._rel_size_2 - 1,
			                                                                        self._embedding_size])])
			question_slot = tf.concat(axis=0, values=[nil_word_slot, self._init([self._vocab_size - 1,
			                                                                     self._embedding_size])])
			# encode entity to vector to calculate weight (Entity Embedding), and do l2 normalization
			self._kg1_entity_embedding = tf.Variable(kg1_entity_slot, name="_kg1_entity_embedding")
			self._kg1_e_norm = tf.nn.l2_normalize(self._kg1_entity_embedding)
			self._kg2_entity_embedding = tf.Variable(kg2_entity_slot, name="_kg2_entity_embedding")
			self._kg2_e_norm = tf.nn.l2_normalize(self._kg2_entity_embedding)
			# encode relation to vector (Relation Embedding), and do l2 normalization
			self._kg1_relation_embedding = tf.Variable(kg1_relation_slot, name="_kg1_relation_embedding")
			self._kg1_r_norm = tf.nn.l2_normalize(self._kg1_relation_embedding)
			self._kg2_relation_embedding = tf.Variable(kg2_relation_slot, name="_kg2_relation_embedding")
			self._kg2_r_norm = tf.nn.l2_normalize(self._kg2_relation_embedding)
			# encode question-words to vector (Question Embedding), and do l2 normalization
			self._question_embedding = tf.Variable(question_slot, name="_question_embedding")
			self._q_norm = tf.nn.l2_normalize(self._question_embedding)
			# Transfer matrices between spaces
			self._kg1_Mrq = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="_kg1_Mrq")
			self._kg1_Mrs = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="_kg1_Mrs")
			self._kg1_Mse = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="_kg1_Mse")
			self._kg1_Mqe = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="_kg1_Mqe")
			self._kg2_Mrq = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="_kg2_Mrq")
			self._kg2_Mrs = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="_kg2_Mrs")
			self._kg2_Mse = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="_kg2_Mse")
			self._kg2_Meq = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="_kg2_Meq")
			# Transfer matrix between two languages
			self._M12 = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="_M12")
		# self._bias = tf.Variable(tf.truncated_normal_initializer([self._embedding_size]))

		# self.GT = tf.Variable(self._init([self._rel_size,1]), name="GT")

		self._nil_vars = {self._kg1_entity_embedding.name, self._kg2_entity_embedding.name,
		                  self._question_embedding.name, self._kg2_relation_embedding.name,
		                  self._kg1_relation_embedding.name}  # need to keep first line 0

	def _kg1_to_train(self):
		"""

        :return: loss for kg1 embedding
        """
		with tf.variable_scope(self._name):
			heads_1 = self._kbs_1[:, 0]  # (batch) head
			relations_1 = self._kbs_1[:, 1]  # (batch) relation
			tails_1 = self._kbs_1[:, 2]  # (batch) tail
			tt = self._padding_1
			kg1_h_matrix = tf.nn.embedding_lookup(self._kg1_entity_embedding, heads_1)  # (batch,e)
			kb1_r_matrix = tf.nn.embedding_lookup(self._kg1_relation_embedding, relations_1)
			kb1_t_matrix = tf.nn.embedding_lookup(self._kg1_entity_embedding, tails_1)
			kg1_tt_matrix = tf.nn.embedding_lookup(self._kg1_entity_embedding, tt)
			kg1_l_matrix = tf.matmul((kg1_h_matrix + kb1_r_matrix), self._kg1_Mse)  # M(h+r)
			kg1_loss_matrix = (kg1_l_matrix - kb1_t_matrix) * (kg1_l_matrix - kb1_t_matrix)
			kg1_neg_matrix = (kg1_l_matrix - kg1_tt_matrix) * (kg1_l_matrix - kg1_tt_matrix)
			kg1_emb_loss = self._margin + tf.reduce_sum(kg1_loss_matrix, 1) - tf.reduce_sum(kg1_neg_matrix, 1)
			kg1_emb_loss = tf.maximum(0.00, kg1_emb_loss)

			return kg1_emb_loss

	def _kg2_to_train(self):
		"""

        :return: loss for kg2 embedding
        """
		with tf.variable_scope(self._name):
			heads_2 = self._kbs_2[:, 0]
			relations_2 = self._kbs_2[:, 1]
			tails_2 = self._kbs_2[:, 2]
			tt = self._padding_2
			kg2_h_matrix = tf.nn.embedding_lookup(self._kg2_entity_embedding, heads_2)  # (batch,e)
			kg2_r_matrix = tf.nn.embedding_lookup(self._kg2_relation_embedding, relations_2)
			kg2_t_matrix = tf.nn.embedding_lookup(self._kg2_entity_embedding, tails_2)
			kg2_tt_matrix = tf.nn.embedding_lookup(self._kg2_entity_embedding, tt)
			kg2_l_matrix = tf.matmul((kg2_h_matrix + kg2_r_matrix), self._kg2_Mse)  # M(h+r)
			kg2_loss_matrix = (kg2_l_matrix - kg2_t_matrix) * (kg2_l_matrix - kg2_t_matrix)
			kg2_neg_matrix = (kg2_l_matrix - kg2_tt_matrix) * (kg2_l_matrix - kg2_tt_matrix)

			kg2_emb_loss = self._margin + tf.reduce_sum(kg2_loss_matrix, 1) - tf.reduce_sum(kg2_neg_matrix, 1)
			kg2_emb_loss = tf.maximum(0.00, kg2_emb_loss)

			return kg2_emb_loss

	def _multi_kg_to_train(self):
		with tf.variable_scope(self._name):
			kg1_entity_ids = self._alignment_seeds[:, 0]
			kg1_entity_emb = tf.nn.embedding_lookup(self._kg1_entity_embedding, kg1_entity_ids)
			kg2_entity_ids = self._alignment_seeds[:, 1]
			kg2_entity_emb = tf.nn.embedding_lookup(self._kg2_entity_embedding, kg2_entity_ids)
			alignment_loss_matrix = tf.subtract(tf.matmul(kg1_entity_emb, self._M12), kg2_entity_emb)
			alignment_loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(alignment_loss_matrix), 1)))
			return alignment_loss

	def _inference(self):
		with tf.variable_scope(self._name):
			# input module
			loss = tf.reshape(self._zeros, [-1, 1], name='loss')  # (batch_size, 1)
			init_state_index = tf.reshape(self._paths[:, 0], [-1, 1])  # (batch_size, 1)
			question_word_embeddings = tf.nn.embedding_lookup(self._question_embedding,
			                                                  self._queries)  # Ax_ij shape is (batch_size,
			# sentence_size ,embedding_size)
			question_emb_1 = tf.reduce_sum(question_word_embeddings, 1)  # shape is (batch_size, embedding_size)
			state_emb = tf.nn.embedding_lookup(self._kg1_entity_embedding, init_state_index)  # (b,1)->(b,1,e)
			state_emb = tf.squeeze(state_emb, [1])  # (batch_size, embedding_size)

			# reasoning module
			reasoning_path = init_state_index
			# (batch_size, embedding_size)*(embedding_size, relations_size) -> (batch_size, relations_size)
			gate_1 = tf.matmul(question_emb_1, tf.matmul(self._kg1_relation_embedding, self._kg1_Mrq),
			                   transpose_b=True) + tf.matmul(state_emb, tf.matmul(self._kg1_relation_embedding,
			                                                                      self._kg1_Mrs), transpose_b=True)

			kg1_relation_logits = gate_1
			kg1_relation_index = tf.argmax(kg1_relation_logits, 1)  # (batch_size, 1)
			gate_1 = tf.nn.softmax(gate_1)  # (batch_size, relations_size)

			# gumble-softmax: gate is unnormalized logits,
			# u = tf.random_uniform(shape=tf.shape(gate),minval=0,maxval=1.0) #(b,r)
			# g = -tf.log(-tf.log(u+1e-20)+1e-20)
			# tau = tf.nn.relu(tf.matmul(gate,self.GT))+1e-8 #(batch,1)
			# gate = tf.nn.softmax((gate) / tau) #(batch,v)

			real_rel_1_one_hot = tf.one_hot(self._paths[:, 1], self._rel_size_1, on_value=1.0, off_value=0.0,
			                                axis=-1)  # (batch_size, relations_size)
			# predict_rel_one_hot = tf.one_hot(r_index, self._rel_size_1, on_value=1.0, off_value=0.0, axis=-1)

			# (batch_size, relations_size) * (relations_size, embedding_size) -> (batch_size, embedding_size)
			state_emb = state_emb + tf.matmul(gate_1, tf.matmul(self._kg1_relation_embedding, self._kg1_Mrs))

			loss += tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=kg1_relation_logits,
			                                                           labels=real_rel_1_one_hot), [-1, 1])  # (b,1)

			question_emb_1 = question_emb_1 - tf.matmul(gate_1, tf.matmul(self._kg1_relation_embedding, self._kg1_Mrq))

			# answer module
			# (entity_size, embedding_size) (batch_size, entities_size)
			kg1_tail_logits = tf.matmul(tf.matmul(state_emb, self._kg1_Mse),
			                            self._kg1_entity_embedding, transpose_b=True)
			# (batch_size, 1)
			kg1_tail_index = tf.argmax(kg1_tail_logits, 1)
			# (batch_size, embedding_size)
			kg1_tail_emb = tf.nn.embedding_lookup(self._kg1_entity_embedding, kg1_tail_index)
			# if r_index == 0, stop inference, ans = previous ans; if not r_index==0, ans = ans
			kg1_tail_index = tf.cast(kg1_tail_index, tf.float32)
			kg1_relation_index = tf.cast(kg1_relation_index, tf.float32)
			kg1_tail_index = kg1_relation_index / (kg1_relation_index + 1e-15) * kg1_tail_index\
			        + (1 - kg1_relation_index / (kg1_relation_index + 1e-15))\
			        * tf.cast(reasoning_path[:, -1], tf.float32)

			reasoning_path = tf.concat(axis=1, values=[reasoning_path,
			                                           tf.reshape(tf.cast(kg1_relation_index, tf.int32), [-1, 1])])
			reasoning_path = tf.concat(axis=1, values=[reasoning_path,
			                                           tf.reshape(tf.cast(kg1_tail_index, tf.int32), [-1, 1])])

			real_kg1_tail_one_hot = tf.one_hot(self._paths[:, 2], self._ent_size_1, on_value=1.0, off_value=0.0,
			                                   axis=-1)  # (b,rel_size)

			loss += tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=kg1_tail_logits,
			                                                           labels=real_kg1_tail_one_hot), [-1, 1])  # (b,1)

			# Cross-lingual Point
			kg2_head_logits = tf.matmul(tf.matmul(kg1_tail_emb, self._M12), self._kg2_entity_embedding, transpose_b=True)
			kg2_head_index = tf.argmax(kg2_head_logits, 1)
			real_kg2_head_one_hot = tf.one_hot(self._paths[:, 3], self._ent_size_2, on_value=1.0, off_value=0.0,
			                                   axis=-1)
			loss += tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=kg2_head_logits,
			                                                           labels=real_kg2_head_one_hot), [-1, 1])
			# Cross-lingual questions
			state_emb = tf.matmul(state_emb, self._M12)
			q_emb_to_ent = tf.matmul(question_emb_1, self._kg1_Mqe)
			question_emb_2 = tf.matmul(tf.matmul(q_emb_to_ent, self._M12), self._kg2_Meq)
			# state_emb = tf.nn.embedding_lookup(self._kg2_entity_embedding, kg2_head_index)
			reasoning_path = tf.concat(axis=1, values=[reasoning_path, tf.reshape(tf.cast(kg2_head_index, tf.int32), [-1, 1])])

			# reasoning for language 2
			gate_2 = tf.matmul(question_emb_2,
			                   tf.matmul(self._kg2_relation_embedding, self._kg2_Mrq),
			                   transpose_b=True) \
			        + tf.matmul(state_emb, tf.matmul(self._kg2_relation_embedding, self._kg2_Mrs), transpose_b=True)
			kg2_relation_logits = gate_2
			kg2_relation_index = tf.argmax(kg2_relation_logits, 1)
			gate_2 = tf.nn.softmax(gate_2)
			real_rel_2_one_hot = tf.one_hot(self._paths[:, 4], self._rel_size_2, on_value=1.0, off_value=0.0,
			                                axis=-1)
			state_emb = state_emb + tf.matmul(gate_2, tf.matmul(self._kg2_relation_embedding, self._kg2_Mrs))

			loss += tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=kg2_relation_logits,
			                                                           labels=real_rel_2_one_hot), [-1, 1])

			# question_emb_2 = question_emb_2 - tf.matmul(gate_2, tf.matmul(self._kg2_relation_embedding, self._kg2_Mrq))

			# answer module
			kg2_tail_logits = tf.matmul(tf.matmul(state_emb, self._kg2_Mse),
			                            self._kg2_entity_embedding, transpose_b=True)
			kg2_tail_index = tf.argmax(kg2_tail_logits, 1)
			# kg2_tail_emb = tf.nn.embedding_lookup(self._kg2_entity_embedding, kg2_tail_index)

			# if r_index == 0, stop inference, ans = previous ans; if not r_index==0, ans = ans
			kg2_tail_index = tf.cast(kg2_tail_index, tf.float32)
			kg2_relation_index = tf.cast(kg2_relation_index, tf.float32)
			kg2_tail_index = kg2_relation_index / (kg2_relation_index + 1e-15) * kg2_tail_index \
			        + (1 - kg2_relation_index / (kg2_relation_index + 1e-15)) \
			        * tf.cast(reasoning_path[:, -1], tf.float32)

			reasoning_path = tf.concat(axis=1, values=[reasoning_path,
			                                           tf.reshape(tf.cast(kg2_relation_index, tf.int32), [-1, 1])])
			reasoning_path = tf.concat(axis=1, values=[reasoning_path,
			                                           tf.reshape(tf.cast(kg2_tail_index, tf.int32), [-1, 1])])

			real_kg2_tail_one_hot = tf.one_hot(self._paths[:, 5], self._ent_size_2, on_value=1.0, off_value=0.0,
			                                   axis=-1)  # (b,rel_size)

			loss += tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=kg2_tail_logits,
			                                                           labels=real_kg2_tail_one_hot), [-1, 1])

		return loss, reasoning_path

	def _embeddings_normalize(self):
		with tf.variable_scope(self._name):
			self._kg1_entity_embedding = self._kg1_e_norm
			self._kg1_relation_embedding = self._kg1_r_norm
			self._kg2_entity_embedding = self._kg2_e_norm
			self._kg2_relation_embedding = self._kg2_r_norm
			self._question_embedding = self._q_norm

	def batch_train_kg1_embedding(self, batch_kb1):
		"""
        Args:
            batch_kb1: Tensor (None, memory_size_1, 3)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
		kg1_n_example = batch_kb1.shape[0]
		pad_1 = np.random.randint(low=0, high=self._ent_size_1, size=kg1_n_example)

		feed_dict = {self._kbs_1: batch_kb1, self._padding_1: pad_1, self._isTrain: 0}

		kg1_loss, _1 = self._sess.run([self.kg1_loss_op, self.kg1_train_op], feed_dict=feed_dict)
		# self._kg1_entity_embedding = tf.nn.l2_normalize(self._kg1_entity_embedding,1)
		# self._kg1_relation_embedding = tf.nn.l2_normalize(self._kg1_relation_embedding,1)
		return kg1_loss

	def batch_train_kg2_embedding(self, batch_kb2):
		"""
        Args:
            batch_kb2: Tensor (None, memory_size_1, 3)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
		kg2_n_example = batch_kb2.shape[0]
		pad_2 = np.random.randint(low=0, high=self._ent_size_2, size=kg2_n_example)

		feed_dict = {self._kbs_2: batch_kb2, self._padding_2: pad_2, self._isTrain: 0}

		kg2_loss, _2 = self._sess.run([self.kg2_loss_op, self.kg2_train_op], feed_dict=feed_dict)
		return kg2_loss

	def batch_train_alignment(self, alignment_seeds):
		feed_dict = {self._alignment_seeds: alignment_seeds}
		loss, _ = self._sess.run([self.alignment_loss_op, self.alignment_train_op], feed_dict=feed_dict)
		return loss

	def batch_train_inference(self, queries, answers, answers_id, paths):
		"""
        Args:
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, ent_size)
            answers_id: Tensor (None, 1)
            paths: Tensor

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
		n_example = queries.shape[0]
		zeros = np.zeros(n_example)
		feed_dict = {self._queries: queries, self._answers: answers,
		             self._answers_id: answers_id, self._paths: paths, self._zeros: zeros, self._isTrain: 0}
		loss, _ = self._sess.run([self.inference_loss_op, self.inference_train_op], feed_dict=feed_dict)
		self._kg1_entity_embedding = self._sess.run(self._kg1_e_norm)
		self._kg1_relation_embedding = self._sess.run(self._kg1_r_norm)
		self._kg2_entity_embedding = self._sess.run(self._kg2_e_norm)
		self._kg2_relation_embedding = self._sess.run(self._kg2_r_norm)
		self._question_embedding = self._sess.run(self._q_norm)
		return loss

	def predict(self, queries, paths):
		"""Predicts answers as one-hot encoding.

        Args:
            queries: Tensor (None, sentence_size)
            paths: Tensor(None, 6)

        Returns:
            answers: id (None, 1)  ,predict_op = max(1, [None,ent_size])
            :param queries:
            :param paths:
        """
		n_example = queries.shape[0]
		zeros = np.zeros(n_example)
		feed_dict = {self._queries: queries, self._paths: paths, self._zeros: zeros, self._isTrain: 1}
		return self._sess.run(self.inference_predict_op, feed_dict=feed_dict)

	def store(self):
		file = os.path.join(self._checkpoint_dir, self._name)
		print(" [*] save current parameters to %s." % file )
		self._saver.save(self._sess, file)

	def load(self):
		print(" [*] Reading checkpoints...")
		checkpoint = tf.train.get_checkpoint_state(self._checkpoint_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			file = os.path.join(self._checkpoint_dir, self._name)
			print("[*] Read from %s" % checkpoint.model_checkpoint_path)
			self._saver.restore(self._sess, checkpoint.model_checkpoint_path)
		else:
			print(" [!] Test mode but no checkpoint found")
		# raise Exception(" [!] Test mode but no checkpoint found")
