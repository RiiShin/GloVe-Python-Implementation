import corpusit
from scipy import sparse
from typing import List, Iterator
import h5py
import multiprocessing
from tqdm import tqdm
import gc
import argparse
import os

cache = {}



def process_chunk(chunk: List[str]):
    vocab = cache['vocab']
    s2i = vocab.s2i_dict()
    size = max(s2i.values()) + 1

    i_list = []
    j_list = []
    co_value = []
    for document in chunk:
        tokens = document.split()
        ids = [s2i.get(token) for token in tokens]
        for cent in range(len(ids)):
            if not ids[cent] and ids[cent] != 0:
                continue

            con_l = max(cent - cache['win_size'], 0)
            con_r = min(cent + cache['win_size'] + 1, len(ids))
            for con in range(con_l, con_r):
                if cent == con or (not ids[con] and ids[con] != 0):
                    continue
                i_list.append(ids[cent])
                j_list.append(ids[con])
                co_value.append(1/abs(cent-con))

    cooc_chunk = sparse.coo_matrix(
        (co_value, (i_list, j_list)), shape=(size, size))
        
    return cooc_chunk.tocsr()



def init_vocab(vocab: corpusit.Vocab, win_size: int):
    cache['vocab'] = vocab
    cache['win_size'] = win_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-cp', "--corpus_path", type=str,
            help='Corpus path.', required=True)
    parser.add_argument('-m', "--min_count", default=100, type=int, 
            help='Threshold of words frequency in vocabulary, default = 100')
    parser.add_argument('-c', "--chunk_size", default=10000, type=int, 
            help='Chunk size in multiprocessing, default = 10000')
    parser.add_argument('-sp', "--save_path", type=str,
            help='Co-occurrence matrix saving name.', required=True)

    args = parser.parse_args()


    # vocab = corpusit.Vocab.from_bin(args.corpus_path)
    vocab = corpusit.Vocab.build(args.corpus_path, min_count=args.min_count)
    print('Vocabulary generated!')
    
    f = open(args.corpus_path, 'r')
    cor_r = (line.lower().strip() for line in f)   # 迭代器, 实时读取.


    def chunker(datait: Iterator, chunk_size):
        chunk = []
        for sample in datait:
            chunk.append(sample)
            if len(chunk) == chunk_size:
                yield chunk[:]
                chunk.clear()
        if len(chunk):
            yield chunk[:]
        return

    chunk_size = 10000
    chunk_iterator = chunker(cor_r, chunk_size=args.chunk_size)   # 生成器, 实时读取

    size = max(vocab.s2i_dict().values()) + 1

    win_size = 10
    with multiprocessing.Pool(32, initializer=init_vocab, initargs=(vocab, win_size)) as pool:
        cooc = sparse.coo_matrix(
            ([], ([], [])), shape=(size, size)).tocsr()
        
        for cooc_chunk in tqdm(pool.imap(process_chunk, chunk_iterator)):
            cooc += cooc_chunk
            # print(f"Not-none entries: {cooc.nnz}")
            gc.collect()

    cooc_coo = cooc.tocoo()


    print('Co-occurrence matrix got!')

    with h5py.File(args.save_path, mode='w') as f:
        f.create_dataset("row", data=cooc_coo.row)
        f.create_dataset("col", data=cooc_coo.col)
        f.create_dataset("data", data=cooc_coo.data)
    f.close()

    print('Co-occurrence matrix hdf5 file saved!')