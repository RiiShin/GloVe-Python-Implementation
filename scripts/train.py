import torch
from scipy.sparse import coo_matrix
import argparse
import random
from glove import GloVe
import h5py
import numpy as np
import tqdm


class dataiterator:
    def __init__(
        self,
        cooc_row: np.ndarray,
        cooc_col: np.ndarray,
        cooc_data: np.ndarray,
        batch_size: int,
    ):
        self.cooc_row = cooc_row
        self.cooc_col = cooc_col
        self.cooc_data = cooc_data
        self.cooc_len = len(cooc_row)
        self.batch_num = (self.cooc_len + batch_size - 1) // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        batch_start_ind = random.randint(0, self.cooc_len - batch_size)
        return (
            torch.LongTensor(
                [
                    self.cooc_row[batch_start_ind : batch_start_ind + batch_size],
                    self.cooc_col[batch_start_ind : batch_start_ind + batch_size],
                ]
            ),
            torch.FloatTensor(
                self.cooc_data[batch_start_ind : batch_start_ind + batch_size]
            ),
        )

    def __len__(self):
        return self.batch_num


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--batch_size",
        default=512,
        type=int,
        help="input batch size please, default = 512",
    )
    parser.add_argument(
        "-cp",
        "--cooc_path",
        type=str,
        help="input co-occurrence matrix saving path!",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--save_path",
        type=str,
        help="input model parameters saving path!",
        required=True,
    )
    parser.add_argument(
        "-ep",
        "--epoch",
        default=15,
        type=int,
        help="input the number of training iteration, default = 15",
    )
    parser.add_argument(
        "-ed",
        "--embed_dim",
        default=100,
        type=int,
        help="input word embedding dimensions, default = 100",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=5e-2,
        type=float,
        help="input initial learning rate, default = 5e-2",
    )
    parser.add_argument(
        "-g",
        "--graph_path",
        type=str,
        help="input curve graph saving path!",
        required=True,
    )

    args = parser.parse_args()

    with h5py.File(args.cooc_path, "r") as f:
        row = f["row"][:]
        col = f["col"][:]
        data = f["data"][:]

    vocab_size = max(row[-1], col[-1]) + 1

    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GloVe(vocab_size, e_dim=args.embed_dim, xmax=100, alpha=0.75).to(
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epoches = args.epoch
    epoch_count = 0
    model.train()
    loss_array = []
    min_loss = 0

    for epoch in range(epoches):
        e_loss = 0
        count = 0
        dataloader = dataiterator(row, col, data, batch_size=batch_size)
        print("Dataloader initialized!")
        for ij, xij in tqdm.tqdm(dataloader):
            ij, xij = ij.cuda(), xij.cuda()
            loss = model(ij[0], ij[1], xij)
            e_loss += loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1

        print(f"Epoch {epoch+1}/{epoches} : loss = {e_loss/count}")
        loss_array.append(e_loss / count)

        if epoch == 0 or min_loss > e_loss / count:
            min_loss = e_loss / count
            torch.save(model.state_dict(), args.save_path)
            epoch_count = 0
        else:
            epoch_count += 1
            if epoch_count >= 2:
                break

    print(f"Minimal loss: {min_loss}")

    print("Training finished!")

    import matplotlib.pyplot as plt

    plx = list(range(1, len(loss_array) + 1))
    ply = loss_array[:]
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(0, len(loss_array) + 1)
    plt.ylim(0, max(ply) * 1.2)
    plt.plot(plx, ply)
    plt.grid(linestyle="-.")
    plt.title("Loss-epoch curve")
    plt.savefig(args.graph_path)

    print("Cruve saved!")
