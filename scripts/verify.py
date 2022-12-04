import numpy as np
import torch.nn as nn
import torch


s2i = {}
s2i['a'] = 0
s2i['b'] = 1
s2i['c'] = 2
print(s2i)

import scipy.sparse as sparse
from typing import Iterator



f = open('./verify_corpus.txt', 'r')
cor_r = (line.lower().strip() for line in f)   # 迭代器, 实时读取.
win_size = 10


def process_chunk(chunk: Iterator):
    size = max(s2i.values()) + 1
    i_list = []
    j_list = []
    co_value = []
    for document in chunk:
        tokens = document.split()
        ids = [s2i.get(token) for token in tokens]
        # for left in range(len(ids)):

        #    for right in range(left, left + cache['win_size']):
        #        cooc_idpair2count[(ids[left], ids[right])] += 1
        for cent in range(len(ids)):
            if not ids[cent] and ids[cent] != 0:
                continue

            con_l = max(cent - win_size, 0)
            con_r = min(cent + win_size + 1, len(ids))
            for con in range(con_l, con_r):
                if cent == con or (not ids[con] and ids[con] != 0):
                    continue
                i_list.append(ids[cent])
                j_list.append(ids[con])
                co_value.append(1/abs(cent-con))

    cooc_chunk = sparse.coo_matrix(
        (co_value, (i_list, j_list)), shape=(size, size))
    # print(cooc_chunk.data[:2])
    return cooc_chunk.tocsr()

    # ...
    # cooc = ...
    # return cooc

cooc = process_chunk(cor_r).tocoo()
print(f'co-occurrence matrix: {cooc}')

class GloVe(nn.Module):

    def __init__(self, v_size: int, e_dim: int, xmax: int, alpha: float):
        super().__init__()
        self.w = nn.Embedding(
            num_embeddings=v_size, 
            embedding_dim=e_dim, sparse = True
        )

        self.w_ = nn.Embedding(
            num_embeddings=v_size, 
            embedding_dim=e_dim, sparse = True
        )

        self.b = nn.Parameter(
            torch.randn(v_size, dtype=torch.float)
        )

        self.b_ = nn.Parameter(
            torch.randn(v_size, dtype=torch.float)
        )
        self.xmax = xmax
        self.alpha = alpha

    def forward(self, i, j, xij):
        loss = torch.sum(torch.mul(self.w(i), self.w_(j)), dim=1)
        loss = torch.square((loss + self.b[i] + self.b_[j] - torch.log(xij)))
        cooc_func = torch.clamp((xij / self.xmax).pow(self.alpha), max = 1)
        loss = torch.mean(torch.mul(cooc_func, loss))
        return loss

length = cooc.nnz
model = GloVe(v_size=3, e_dim=3, xmax=3, alpha=0.75)


loss = model(torch.LongTensor(cooc.row), torch.LongTensor(cooc.col), torch.FloatTensor(cooc.data))
# print(model.w(torch.LongTensor([0,1,2])))
# print(model.w_(torch.LongTensor([0,1,2])))
# print(model.b)
# print(model.b_)
# print(loss)

nw, nw_ = model.w(torch.LongTensor([0,1,2])).detach().numpy(), model.w_(torch.LongTensor([0,1,2])).detach().numpy()
nb, nb_ = model.b.detach().numpy(), model.b_.detach().numpy()
nw, nw_ = nw.astype(np.float32), nw_.astype(np.float32)
nb, nb_ = nb.astype(np.float32), nb_.astype(np.float32)

def np_loss(w: np.ndarray, w_: np.ndarray, b: np.ndarray, b_: np.ndarray):
    loss_array = []
    for k in range(length):
        loss_array.append(np.dot(nw[cooc.row[k]].T, nw_[cooc.col[k]]) + nb[cooc.row[k]] + nb_[cooc.col[k]])

    for k in range(length):
        loss_array[k] -= np.log(cooc.data[k].astype(np.float32))
        loss_array[k] *= loss_array[k]
    
    def x_func(xij: np.ndarray, xmax: float, alpha: float):
        return np.power(xij/xmax, alpha)

    loss_array *= x_func(cooc.data.astype(np.float32), 3., 0.75)
    return np.mean(loss_array)


def test_func():
    assert format(float(np_loss(nw, nw_, nb, nb_)), '.6f') == format(float(loss.detach().numpy()), '.6f')