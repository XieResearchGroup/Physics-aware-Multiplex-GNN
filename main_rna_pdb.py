import os
import os.path as osp
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader

from models import PAMNet, Config
from datasets import RNAPDBDataset
from utils import Sampler
from losses import p_losses

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False

def test(model, loader, device):
    model.eval()

    pred_list = []
    y_list = []

    for data in loader:
        data = data.to(device)
        pred = model(data)
        pred_list += pred.reshape(-1).tolist()
        y_list += data.y.reshape(-1).tolist()

    pred = np.array(pred_list).reshape(-1,)
    pred = torch.tensor(pred).to(device)

    y = np.array(y_list).reshape(-1,)
    y = torch.tensor(y).to(device)

    loss = F.smooth_l1_loss(pred, y)
    return loss.item(), np.array(pred_list).reshape(-1,)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=40, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='RNA-Puzzles', help='Dataset to be used')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss).')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=64, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--cutoff_l', type=float, default=0.26, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=2.00, help='cutoff in global layer')
    parser.add_argument('--timesteps', type=int, default=500, help='timesteps')
    args = parser.parse_args()
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    # Creat dataset
    path = osp.join('.', 'data', args.dataset)
    train_dataset = RNAPDBDataset(path, name='train-pkl').shuffle()
    val_dataset = RNAPDBDataset(path, name='val-pkl')

    # Load dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print("Data loaded!")
    for data, name in train_loader:
        print(data)
        break

    sampler = Sampler(timesteps=args.timesteps)
    config = Config(dataset=args.dataset, dim=args.dim, n_layer=args.n_layer, cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g)

    model = PAMNet(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("Start training!")
    best_val_loss = None
    for epoch in range(args.epochs):
        model.train()

        for data, name in train_loader:
            losses = []
            data = data.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, args.timesteps, (data.batch.shape[0],), device=device).long() # Generate random timesteps

            loss = p_losses(model, data, t, sampler=sampler, loss_type="huber")

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # train_loss, _ = test(model, train_loader, device)
        # val_loss, _ = test(model, val_loader, device)

        # print('Epoch: {:03d}, Train Loss: {:.7f}, Val Loss: {:.7f}'.format(epoch+1, train_loss, val_loss))
        print(f'Epoch: {epoch+1}, Loss: {np.mean(losses):.4f}')
        
        # save_folder = os.path.join(".", "save", args.dataset)
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)

        # if best_val_loss is None or val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), os.path.join(save_folder, "best_model.h5"))


if __name__ == "__main__":
    main()