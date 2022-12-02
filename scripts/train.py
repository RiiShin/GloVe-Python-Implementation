import torch
from scipy.sparse import coo_matrix
import argparse
import random
from glove import GloVe
import h5py
import numpy as np


def dataiterator(cooc_row: np.ndarray, cooc_col: np.ndarray, cooc_data: np.ndarray, batch_size: int):
    cooc_len = len(cooc_row)
    batch_num = cooc_len // batch_size + 1
    for i in range(batch_num):
        batch_ind = [random.randint(0, cooc_len-1) for _ in range(batch_size)]
        yield torch.LongTensor([[cooc_row[k] for k in batch_ind], [cooc_col[k] for k in batch_ind]]), torch.FloatTensor([cooc_data[k] for k in batch_ind])
    
    return




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-b', "--batch_size", default=512, type=int, 
            help='input batch size please, default = 512')
    parser.add_argument('-p', "--save_path", type=str,
            help='input model parameters saving path!', required=True)
    parser.add_argument('-ep', "--epoch", default=15, type=int, 
            help='input the number of training iteration, default = 15')
    parser.add_argument('-ed', "--embed_dim", default=100, type=int, 
            help='input word embedding dimensions, default = 100')
    parser.add_argument('-lr', "--learning_rate", default=5e-2, type=float, 
            help='input initial learning rate, default = 5e-2')

    args = parser.parse_args()

    with h5py.File('./cooc_matrix/h5_cooc_matrix.hdf5', 'r') as f:
        row = f['row'][:]
        col = f['col'][:]
        data = f['data'][:]
    f.close
    
    vocab_size = max(row[-1], col[-1]) + 1


    batch_size = args.batch_size
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GloVe(vocab_size, e_dim=args.embed_dim, xmax=100, alpha=0.75).to(device=device)



    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
    epoches = args.epoch
    epoch_count = 0
    model.train()
    loss_array = []
    min_loss = 0


    for epoch in range(epoches):
        e_loss = 0
        count = 0
        dataloader = dataiterator(row, col, data, batch_size=batch_size)
        print('Dataloader initialized!')
        for ij, xij in dataloader:
            ij, xij = ij.cuda(), xij.cuda()
            loss = model(ij[0], ij[1], xij)
            e_loss += loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1

        print(f"Epoch {epoch+1}/{epoches} : loss = {e_loss/count}")    
        loss_array.append(e_loss/count)

        if epoch == 0 or min_loss > e_loss/count:
            min_loss = e_loss/count
            torch.save(model.state_dict(), args.save_path)
            epoch_count = 0
        else:
            epoch_count += 1
            if epoch_count >= 2:
                break

    print(f'Minimal loss: {min_loss}')

    print('Training finished!')