import torch
import torch.nn.functional as F
from utils import Sampler

def p_losses(denoise_model,
             x_data,
             t,
             sampler: Sampler,
             loss_type="huber",
             noise=None,
             ):

    x_pos = x_data.x[:, :3]  # Get the position of the atoms. First 3 features are the coordinates
    if noise is None:
        noise = torch.randn_like(x_pos)
    x_noisy = sampler.q_sample(x_start=x_pos,
                               t=t,
                               noise=noise,
                               )
    x_data.x[:, :3] = x_noisy  # Replace the position of the atoms with the noisy data
    predicted_noise = denoise_model(x_data, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss