# coding: utf-8
"""Process input data.

Knowledge Base: h@@@r@@@t

Training/Testing data: question@@@answer@@@e1(l1)#r1(l1)#e2(l1)#e2(l2)#r2(l2)#e3(l2)
Note that the last item of each line should be an entity.
"""
from __future__ import absolute_import

import os
import numpy as np
import jieba


def read_knowledge_base_file(knowledge_base_file, separator="@@@"):
	"""
Reads a knowledge base file and saves all the entities and relations
to the passed parameters.
Example in knowledge_base_file: batch_kb1.txt h@@@r@@@t
"""
	rel2id = {'<end>': 0}
	ent2id = {'<unk>': 0}
	print("Reading knowledge base file...")
	if os.path.isfile(knowledge_base_file):
		with open(knowledge_base_file, encoding="utf-8") as f:
			lines = f.readlines()
	else:
		raise Exception("!! %s is not found!!" % knowledge_base_file)

	for line in lines:
		line = line.strip().split(separator)
		if line[0] not in ent2id.keys():
			ent2id[line[0]] = len(ent2id)
		if line[2] not in ent2id.keys():
			ent2id[line[2]] = len(ent2id)
		if line[1] not in rel2id.keys():
			rel2id[line[1]] = len(rel2id)
	return ent2id, rel2id


def get_knowledge_base(knowledge_base_file, ent2id, rel2id, separator="@@@"):
	"""
Create the KB Matrix whose elements are integers representing the knowledge base.
    :param separator: used to split a line of data
:param knowledge_base_file: Knowledge base file whose triples are separated by separator.
:param ent2id: Entity to ID dictionary.
:param rel2id: Relation to ID dictionary.
:return: Triples Numpy Array, KB Matrix whose maximum column index is the maximum number of tails for a single
head entity and relation pair, maximum tail number.
"""
	n_entities = len(ent2id)
	n_relations = len(rel2id)
	tails = np.zeros([n_entities * n_relations, 1], 'int32')
	triples = []

	with open(knowledge_base_file, encoding="utf-8") as f:
		for line in f.readlines():
			line = line.strip().split(separator)

			h = ent2id[line[0]]
			r = rel2id[line[1]]
			t = ent2id[line[2]]
			triples.append([h, r, t])  # [h,r]->[h*n_relations+r]
			tails[h * n_relations + r] += 1

	return np.array(triples), np.max(tails)


def read_dataset(data_file, separator="@@@", is_cn=False):
	"""
question@@@answer@@@e1(l1)#r1(l1)#e2(l1)#e2(l2)#r2(l2)#e3(l2)
"""
	if os.path.isfile(data_file):
		with open(data_file, encoding="utf-8") as f:
			lines = f.readlines()
	else:
		raise Exception("!! %s is not found!!" % data_file)

	data = []
	questions = []
	word2id = {'<unk>': 0}
	if is_cn:
		jieba.load_userdict("data/zh_ent_dict.txt")
	for line in lines:
		line = line.strip().split(separator)

		if is_cn:
			line[0] = " ".join(jieba.cut(line[0], cut_all=True))  # Use jieba to do word cut

		q_words_list = line[0].strip().split()
		data.append([line[0], line[1], line[2]])
		for w in q_words_list:
			if w not in word2id.keys():
				word2id[w] = len(word2id)
		questions.append(q_words_list)

	sentence_size = max(len(i) for i in questions)

	return data, sentence_size, word2id


class KnowledgeBase(object):

	def __init__(self, knowledge_base_file, separator="@@@", name="kb"):
		self.name = name

		print("Processing knowledge base for " + name + "...")
		ent2id, rel2id = read_knowledge_base_file(knowledge_base_file, separator)
		self.entities = ent2id.keys()
		self.relations = rel2id.keys()
		self.rel2id = rel2id
		self.ent2id = ent2id
		self.n_entities = len(self.ent2id)
		self.n_relations = len(self.rel2id)

		print('here are %d relations in rel2id(relation_vocabulary)' % self.n_relations)
		print('here are %d entities in ent2id(entity_vocabulary)' % self.n_entities)

		self.triples, self.tails_size = get_knowledge_base(knowledge_base_file, self.ent2id, self.rel2id)

		print("#number of Triples", len(self.triples))


class MultiKnowledgeBase(object):

	def __init__(self, kb1: KnowledgeBase, kb2: KnowledgeBase, alignment_file: str, separator="@@@"):
		self.align_dict_1_2 = {}
		self.align_dict_2_1 = {}
		self.align_array = []
		self.n_align = 0
		self.kb1 = kb1
		self.kb2 = kb2
		self.name = self.kb1.name + "+" + self.kb2.name
		with open(alignment_file, "r", encoding="utf-8", ) as af:
			for line in af.readlines():
				line = line.strip().split(separator)
				e1 = None
				e2 = None
				if line[0] in self.kb1.ent2id.keys():
					e1 = self.kb1.ent2id[line[0]]
				if line[1] in self.kb2.ent2id.keys():
					e2 = self.kb2.ent2id[line[1]]
				if e1 is not None and e2 is not None:
					self.align_array.append((e1, e2))
					self.align_dict_1_2[e1] = e2
					self.align_dict_2_1[e2] = e1
					self.n_align += 1
		self.align_array = np.array(self.align_array)
		print("Loaded aligned entities from", alignment_file, ". #pairs:", self.n_align)


def process_dataset(data_file, multi_kb: MultiKnowledgeBase, separator="@@@", is_cn=False):
	data, sentence_size, word2id = read_dataset(data_file, separator, is_cn)
	print('here are %d words in word2id(question vocab)' % len(word2id))
	kb2_ent2id = ans2id = multi_kb.kb2.ent2id
	kb1_ent2id = multi_kb.kb1.ent2id
	kb1_rel2id = multi_kb.kb1.rel2id
	kb2_rel2id = multi_kb.kb2.rel2id
	questions_ids = []
	questions_strings = []
	answers_id_one_hot = []
	answers_id = []
	paths_ids = []
	paths_strings = []
	for q_str, a_str, p_str in data:
		q_strings = q_str.strip().split()
		questions_ids.append([word2id[w] for w in q_strings] + [0] * max(0, sentence_size - len(q_strings)))
		questions_strings.append(q_strings)
		# answer id one-hot
		a_one_hot = np.zeros(len(ans2id))
		a_one_hot[ans2id[a_str]] = 1
		answers_id_one_hot.append(a_one_hot)
		answers_id.append(ans2id[a_str])
		p_str = p_str.strip().split("#")
		paths_strings.append(p_str)
		paths_ids.append([kb1_ent2id[p_str[0]], kb1_rel2id[p_str[1]], kb1_ent2id[p_str[2]], kb2_ent2id[p_str[3]],
		                  kb2_rel2id[p_str[4]], kb2_ent2id[p_str[5]]])
	return np.array(questions_ids), np.array(answers_id_one_hot), np.array(paths_ids), questions_strings, answers_id, \
	       paths_strings, sentence_size, word2id, len(word2id)
