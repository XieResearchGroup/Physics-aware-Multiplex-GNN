import os
import os.path as osp
import argparse
import numpy as np
import pandas as pd
import random
import torch
from torch_geometric.data import DataLoader

from models import PAMNet, Config
from datasets import TUDataset

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=40, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='rna_native', help='Dataset to be used')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss).')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=16, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--cutoff_l', type=float, default=2.6, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=20.0, help='cutoff in global layer')
    parser.add_argument('--flow', type=str, default='target_to_source', help='Flow direction of message passing')
    parser.add_argument('--saved_model', type=str, default='pamnet_rna.pt', help='Saved model for inference')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    set_seed(args.seed)

    # Creat dataset
    path = osp.join('.', 'data', 'RNA-Puzzles')
    test_dataset = TUDataset(path, name=args.dataset, use_node_attr=True)

    # Load dataset
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("Data loaded!")

    config = Config(dataset=args.dataset, dim=args.dim, n_layer=args.n_layer, cutoff_l=args.cutoff_l, 
                    cutoff_g=args.cutoff_g, flow=args.flow)

    model = PAMNet(config).to(device)
    model.load_state_dict(torch.load("./save/" + args.saved_model, map_location=device))
    model.eval()
    
    print("Model loaded. Start prediction!")
    y_hat_list = []
    df = pd.DataFrame()

    for data in test_loader:
        data = data.to(device)
        output = model(data)
        y_hat_list += output.reshape(-1).tolist()

    y_hat = np.array(y_hat_list).reshape(-1,)

    name_list = np.loadtxt(osp.join('.', 'data', 'RNA-Puzzles', args.dataset, 'raw', args.dataset + '_graph_names.txt'), dtype=str, converters = {0: lambda s: s[:-4]})

    df['PAMNet'] = y_hat
    df['tag'] = name_list
    df['puzzle_number'] = args.dataset[5:]

    if not os.path.exists(osp.join('.', 'rna_puzzles_predictions')):
        os.makedirs(osp.join('.', 'rna_puzzles_predictions'))

    file_name = osp.join('.', 'rna_puzzles_predictions', args.dataset + '.csv')
    df.to_csv(file_name, sep=',', index=False)
    
    print("Prediction saved.")


if __name__ == "__main__":
    main()