import os
import os.path as osp
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
import wandb

from models import PAMNet, Config
from datasets import RNAPDBDataset
from utils import Sampler, SampleToPDB
from losses import p_losses

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(model, loader, device, sampler, args):
    model.eval()
    losses = []
    denoise_losses = []
    for data, name in loader:
        data = data.to(device)
        t = torch.randint(0, args.timesteps, (args.batch_size,), device=device).long() # Generate random timesteps
        graphs_t = t[data.batch]
        loss, denoise_loss = p_losses(model, data, graphs_t, sampler=sampler, loss_type="huber")
        losses.append(loss.item())
        denoise_losses.append(denoise_loss.item())
    return np.mean(losses), np.mean(denoise_losses)

def sample(model, loader, device, sampler, epoch, num_batches=None, exp_name: str = "run"):
    model.eval()
    s = SampleToPDB()
    s_counter = 0
    for data, name in loader:
        data = data.to(device)
        samples = sampler.sample(model, data)[-1]
        s.to('xyz', samples, f"./samples/{exp_name}/{epoch}", name)
        try:
            s.to('trafl', samples, f"./samples/{exp_name}/{epoch}", name)
            s_counter += 1
        except ValueError:
            print("Cannot save molecules with missing P atom.")

        if num_batches is not None and s_counter >= num_batches:
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=40, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='RNA-Puzzles', help='Dataset to be used')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=64, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--cutoff_l', type=float, default=0.5, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=1.60, help='cutoff in global layer')
    parser.add_argument('--timesteps', type=int, default=500, help='timesteps')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--mode', type=str, default='coarse-grain', help='Mode of the dataset')
    parser.add_argument('--knns', type=int, default=2, help='Number of knn neighbors')
    args = parser.parse_args()
    
    if args.wandb:
        wandb.login()
        run = wandb.init(project='RNA-GNN-Diffusion', config=args)
        exp_name = run.name
    else:
        exp_name = "test"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)
    print("Device: ", device)

    # Creat dataset
    path = osp.join('.', 'data', args.dataset)
    # train_dataset = RNAPDBDataset(path, name='train-raw-pkl', mode=args.mode).shuffle()
    train_dataset = RNAPDBDataset(path, name='desc-pkl', mode=args.mode).shuffle()
    val_dataset = RNAPDBDataset(path, name='val-raw-pkl', mode=args.mode)
    samp_dataset = RNAPDBDataset(path, name='val-raw-pkl', mode=args.mode)

    # Load dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    samp_loader = DataLoader(samp_dataset, batch_size=6, shuffle=False)
    print("Data loaded!")
    for data, name in train_loader:
        print(data)
        break

    sampler = Sampler(timesteps=args.timesteps)
    config = Config(dataset=args.dataset, dim=args.dim, n_layer=args.n_layer, cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g, mode=args.mode, knns=args.knns)

    model = PAMNet(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # model_path = f"save/divine-shadow-186/model_305.h5"
    # model.load_state_dict(torch.load(model_path))
    model.to(device)
    print("Start training!")
    
    for epoch in range(args.epochs):
        model.train()
        step = 0
        losses = []
        denoise_losses = []
        for data, name in train_loader:
            
            data = data.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, args.timesteps, (args.batch_size,), device=device).long() # Generate random timesteps
            graphs_t = t[data.batch]
            
            loss_all, loss_denoise = p_losses(model, data, graphs_t, sampler=sampler, loss_type="huber")

            loss_all.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) # prevent exploding gradients
            optimizer.step()
            losses.append(loss_all.item())
            denoise_losses.append(loss_denoise.item())
            if step % 200 == 0 and step != 0 and args.wandb:
                val_loss, val_denoise_loss = test(model, val_loader, device, sampler, args)
                print(f'Epoch: {epoch+1}, Step: {step}, Loss: {np.mean(losses):.4f}, Denoise Loss: {np.mean(denoise_losses):.4f}, Val Loss: {val_loss:.4f}, Val Denoise Loss: {val_denoise_loss:.4f}')
                wandb.log({'Train Loss': np.mean(losses), 'Val Loss': val_loss, 'Denoise Loss': np.mean(denoise_losses), 'Val Denoise Loss': val_denoise_loss,})
                losses = []
                denoise_losses = []
            elif not args.wandb:
                print(f"Epoch: {epoch}, step: {step}, loss: {loss_all.item():.4f} ")
                # val_loss, val_denoise_loss = test(model, val_loader, device, sampler, args)
                # print(f'Val Loss: {val_loss:.4f}, Val Denoise Loss: {val_denoise_loss:.4f}')
            step += 1
        
        val_loss, val_denoise_loss = test(model, val_loader, device, sampler, args)


        if args.wandb:
            wandb.log({'Train Loss': np.mean(losses), 'Val Loss': val_loss, 'Denoise Loss': np.mean(denoise_losses), 'Val Denoise Loss': val_denoise_loss,})
        print(f'Epoch: {epoch+1}, Loss: {np.mean(losses):.4f}, Denoise Loss: {np.mean(denoise_losses):.4f}, Val Loss: {val_loss:.4f}, Val Denoise Loss: {val_denoise_loss:.4f}')
        
        if epoch % 5 == 0:
            sample(model, samp_loader, device, sampler, epoch=epoch, num_batches=1, exp_name=exp_name)
        
        save_folder = f"./save/{exp_name}"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if epoch %1 == 0:
            torch.save(model.state_dict(), f"{save_folder}/model_{epoch}.h5")

        # if best_val_loss is None or val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), os.path.join(save_folder, "best_model.h5"))
    torch.save(model.state_dict(), f"{save_folder}/model_{epoch}.h5")

if __name__ == "__main__":
    main()