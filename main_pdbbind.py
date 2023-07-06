import os.path as osp
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

from models import PAMNet, Config
from utils import rmse, mae, sd, pearson
from datasets import TUDataset


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    y = np.array(y_list).reshape(-1,)
    return rmse(y, pred), mae(y, pred), sd(y, pred), pearson(y, pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=805, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='PDBbind', help='Dataset to be used')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss).')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=128, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--cutoff_l', type=float, default=2.0, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=6.0, help='cutoff in global layer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    # Creat dataset
    path = osp.join('.', 'data', args.dataset)
    refined_dataset = TUDataset(path, name='train_val', use_node_attr=True).shuffle()
    core_dataset = TUDataset(path, name='test', use_node_attr=True)

    # Split dataset
    idx_train, idx_val = train_test_split(np.arange(len(refined_dataset)), test_size=0.1, shuffle=False, random_state=args.seed)

    train_dataset = refined_dataset[torch.LongTensor(idx_train)]
    val_dataset = refined_dataset[torch.LongTensor(idx_val)]
    test_dataset = core_dataset

    # Load dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("Data loaded!")

    config = Config(dataset=args.dataset, dim=args.dim, n_layer=args.n_layer, cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g)

    model = PAMNet(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
    scheduler = MultiStepLR(optimizer, milestones=[50,100,150,200,250,300,350,400,450,500], gamma=0.2)

    print("Start training!")
    best_val_rmse = None
    for epoch in range(args.epochs):
        model.train()

        for data in train_loader:  
            data = data.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = F.mse_loss(output, data.y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        train_rmse, train_mae, train_sd, train_p = test(model, train_loader, device)
        val_rmse, val_mae, val_sd, val_p = test(model, val_loader, device)

        if best_val_rmse is None or val_rmse < best_val_rmse:
            test_rmse, test_mae, test_sd, test_p = test(model, test_loader, device)
            best_val_rmse = val_rmse

        print('Epoch: {:03d}, Train RMSE: {:.7f}, Train MAE: {:.7f}, Train SD: {:.7f}, Train P: {:.7f}, \
            Test RMSE: {:.7f}, Test MAE: {:.7f}, Test SD: {:.7f}, Test P: {:.7f}'.format(epoch+1, train_rmse, train_mae, train_sd, train_p,
                                                                                    test_rmse, test_mae, test_sd, test_p))

    print('Testing RMSE:', test_rmse)
    print('Testing MAE:', test_mae)
    print('Testing SD:', test_sd)
    print('Testing P:', test_p)


if __name__ == "__main__":
    main()