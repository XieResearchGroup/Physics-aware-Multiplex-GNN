import os
import os.path as osp
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from models import PAMNet, PAMNet_s, Config
from utils import EMA
from datasets import QM9


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(model, loader, ema, device):
    mae = 0
    ema.assign(model)
    for data in loader:
        data = data.to(device)
        output = model(data)
        mae += (output - data.y).abs().sum().item()
    ema.resume(model)
    return mae / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=480, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='QM9', help='Dataset to be used')
    parser.add_argument('--model', type=str, default='PAMNet', choices=['PAMNet', 'PAMNet_s'], help='Model to be used')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss).')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=128, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--target', type=int, default="7", help='Index of target for prediction')
    parser.add_argument('--cutoff_l', type=float, default=5.0, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=5.0, help='cutoff in global layer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)
    
    class MyTransform(object):
        def __call__(self, data):
            target = args.target
            if target in [7, 8, 9, 10]:
                target = target + 5
            data.y = data.y[:, target]
            return data

    # Creat dataset
    path = osp.join('.', 'data', args.dataset)
    dataset = QM9(path, transform=MyTransform()).shuffle()

    # Split dataset
    train_dataset = dataset[:110000]
    val_dataset = dataset[110000:120000]
    test_dataset = dataset[120000:]

    # Load dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print("Data loaded!")

    config = Config(dataset=args.dataset, dim=args.dim, n_layer=args.n_layer, cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g)

    if args.model == 'PAMNet':
        model = PAMNet(config).to(device)
    else:
        model = PAMNet_s(config).to(device)
    print("Number of model parameters: ", count_parameters(model))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

    ema = EMA(model, decay=0.999)

    print("Start training!")
    best_val_loss = None
    for epoch in range(args.epochs):
        loss_all = 0
        step = 0
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = F.l1_loss(output, data.y)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)
            optimizer.step()

            curr_epoch = epoch + float(step) / (len(train_dataset) / args.batch_size)
            scheduler_warmup.step(curr_epoch)

            ema(model)
            step += 1
        loss = loss_all / len(train_loader.dataset)
        val_loss = test(model, val_loader, ema, device)

        save_folder = osp.join(".", "save", args.dataset)
        if not osp.exists(save_folder):
            os.makedirs(save_folder)

        if best_val_loss is None or val_loss <= best_val_loss:
            test_loss = test(model, test_loader, ema, device)
            best_val_loss = val_loss
            torch.save(model.state_dict(), osp.join(save_folder, "best_model.h5"))

        print('Epoch: {:03d}, Train MAE: {:.7f}, Val MAE: {:.7f}, '
            'Test MAE: {:.7f}'.format(epoch+1, loss, val_loss, test_loss))
    print('Best Validation MAE:', best_val_loss)
    print('Testing MAE:', test_loss)


if __name__ == "__main__":
    main()