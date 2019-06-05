import torch
import numpy as np
SOS_token = 0
EOS_token = 1
PAD_token = 2
use_cuda = torch.cuda.is_available()

def binaryMatrix(l, value):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m




class AverageMeter(object):
    """
    Computes and stores the average and current value
    Borrowed from ImageNet training in PyTorch project
    https://github.com/pytorch/examples/tree/master/imagenet
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_word_embeddings(embedding_size, pre_trained_embeddings, vocab):
    words = []
    vectors = []
    word2idx = {}

    with open(pre_trained_embeddings, 'r') as f:
        idx = 0
        for l in f:
            line = l.strip().split(" ")
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array([i.replace(",",".") for i in line[1:]]).astype(np.float)
            vectors.append(vect)

    glove = {w: vectors[word2idx[w]] for w in words}
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, embedding_size))
    words_found = 0

    for key, value in vocab.items():
        try:
            weights_matrix[value] = glove[key]
            words_found += 1
        except KeyError:
            weights_matrix[value] = np.random.normal(scale=0.6, size=(embedding_size,))
    return weights_matrix



def tensor2np(tensor):
    return tensor.data.cpu().numpy()