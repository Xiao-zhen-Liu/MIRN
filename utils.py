# coding: utf-8
import sys, multiprocessing, time
from numpy import linalg as LA
import heapq as HP
import numpy as np
import tensorflow as tf
from sklearn import metrics
from data_utils import MultiKnowledgeBase
from multiprocessing import Process, Value, Lock, Manager


def norm(matrix):
    '''Matrix normalization'''
    n = tf.sqrt(tf.reduce_sum(matrix * matrix, 1))
    return tf.reshape(n, [-1, 1])


def MatrixCos(inputdata, key):
    '''inputdata = [batch,embed]
        key = [slot,embed]
    return most similar key_id for each inputdata'''
    addressing = tf.matmul(inputdata, key, transpose_b=True)  # (b,e)*(e,slots) -> (b,s)
    norm1 = norm(inputdata)  # (b,1)
    norm2 = norm(key)  # (s,1)
    n = tf.matmul(norm1, norm2, transpose_b=True) + 1e-8  # (b,s)
    addressing = tf.div(addressing, n)
    index = tf.reshape(tf.argmax(addressing, 1), [-1, 1])  # (b,1)
    return tf.to_int32(index)


def SimpleMatrixCos(inputdata, key):
    inputdata = tf.nn.l2_normalize(inputdata, 1)
    key = tf.nn.l2_normalize(key, 1)
    addressing = tf.matmul(inputdata, key, transpose_b=True)  # (b,4)*(4,5) -> (b,5)
    index = tf.reshape(tf.argmax(addressing, 1), [-1, 1])  # (b,1)
    return tf.to_int32(index)


def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
        m_i = sum_j l_ij*A*x_ij /J/d
        l_ij = Jd-jd-iJ+2ij  = ij-Ji/2-jd/2+Jd/4
    return l-matrix-transpose (fixed)
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = (1 + 4 * encoding / embedding_size / sentence_size) / 2
    return np.transpose(encoding)


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.
    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))  # tf.zeros([1,s])
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)


def multi_accuracy(labels: np.ndarray, predictions: np.ndarray, multi_base: MultiKnowledgeBase, steps: list,
                   hops, lan_labels):
    # compare path and final answer accuracy
    accuracies = []

    for i in range(steps[-1]+1):
        accuracies.append(round(metrics.accuracy_score(labels[:, i], predictions[:, i]), 4))

    align_dict_1_2 = multi_base.align_dict_1_2
    align_dict_2_1 = multi_base.align_dict_2_1
    align_accuracies = "Alignment accuracy on reasoning:"
    for i in range(hops - 1):
        if lan_labels[i + 1] != lan_labels[i]:
            if lan_labels[i + 1] == lan_labels[0]:
                anf = 0
                alignment_labels = []
                for ent_id in predictions[:, steps[i+1]].tolist():
                    if ent_id in align_dict_2_1.keys():
                        alignment_labels.append(align_dict_2_1[ent_id])
                    else:
                        alignment_labels.append(0)
                        anf += 1
                aligned_ids = predictions[:, steps[i+1]+1].tolist()
                l = len(alignment_labels)
                align_accuracies += "\nAlign miss rate 2-1: " + str(round(anf / l, 4))
                a = 0
                for i in range(l):
                    if alignment_labels[i] != 0:
                        if alignment_labels[i] == aligned_ids[i]:
                            a += 1
                    else:
                        l -= 1
                align_accuracies += ", Align accuracy 2-1: " + (str(round(a / l, 4)) if l > 0 else str(0))
            else:
                anf = 0
                alignment_labels = []
                for ent_id in predictions[:, steps[i+1]].tolist():
                    if ent_id in align_dict_1_2.keys():
                        alignment_labels.append(align_dict_1_2[ent_id])
                    else:
                        alignment_labels.append(0)
                        anf += 1
                aligned_ids = predictions[:, steps[i+1]+1].tolist()
                l = len(alignment_labels)
                align_accuracies += "\nAlign miss rate 1-2: " + str(round(anf / l, 4))
                a = 0
                for i in range(l):
                    if alignment_labels[i] != 0:
                        if alignment_labels[i] == aligned_ids[i]:
                            a += 1
                    else:
                        l -= 1
                align_accuracies += ", Align accuracy 1-2: " + (str(round(a / l, 4)) if l > 0 else str(0))
    return accuracies, align_accuracies

def create_batch(instances, batch_size):
    s = list(zip(range(0, instances - batch_size, batch_size), range(batch_size, instances, batch_size)))
    s.append(((instances - instances % batch_size, instances)))
    return s


def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)


def orthogonal_initializer(scale=1.0, dtype=tf.float32):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)
  return _initializer


def NN(vec, vec_pool):
    min_dist = sys.maxsize
    rst = None
    for i in range(len(vec_pool)):
        dist = LA.norm(vec - vec_pool[i], ord=1)
        if dist < min_dist:
            min_dist = dist
            rst = i
    return rst


def kNN(vec, vec_pool, topk=10):
    q = []
    for i in range(len(vec_pool)):
        dist = LA.norm(vec - vec_pool[i], ord=1)
        if len(q) < topk:
            HP.heappush(q, self.index_dist(i, dist))
        else:
            #indeed it fetches the biggest
            tmp = HP.nsmallest(1, q)[0]
            if tmp.dist > dist:
                HP.heapreplace(q, self.index_dist(i, dist) )
    rst = []
    while len(q) > 0:
        item = HP.heappop(q)
        rst.insert(0, (item.index, item.dist))
    return rst

def align_predict(a_12_vecs, a_21_vecs, e_emb_1, e_emb_2, alignments, index, a_12_ids, a_21_ids, t0):
    while index.value < len(alignments):
        id = index.value
        index.value += 1
        if id % 100 == 0:
            print("Tested %d in %d seconds." % (id, time.time()-t0))
            print(metrics.accuracy_score(alignments[:id, 1], a_12_ids[:id]),
                  metrics.accuracy_score(alignments[:id, 0], a_21_ids[:id]))

        a_12_ids.append(NN(a_12_vecs[id], e_emb_2))
        a_21_ids.append(NN(a_21_vecs[id], e_emb_1))

def parallel_predict(a_12_vecs, a_21_vecs, e_emb_1, e_emb_2, alignments):
    manager = Manager()
    index = Value('i', 0, lock=True)  # index
    a_12_ids = manager.list()
    a_21_ids = manager.list()
    t0 = time.time()
    cpu_count = multiprocessing.cpu_count()
    processes = [Process(target=align_predict(a_12_vecs, a_21_vecs, e_emb_1, e_emb_2, alignments, index, a_12_ids,
                                              a_21_ids, t0)) for x in range(cpu_count - 1)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    return metrics.accuracy_score(alignments[:, 1], a_12_ids), metrics.accuracy_score(alignments[:, 0], a_21_ids)
