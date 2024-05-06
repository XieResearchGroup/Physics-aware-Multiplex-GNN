import torch
import torch.nn.functional as F
from utils import Sampler, generate_per_residue_noise

def p_losses(denoise_model,
             x_data,
             t,
             sampler: Sampler,
             loss_type="huber",
             noise=None,
             ):

    x_start = x_data.x.contiguous()  # Get the position of the atoms. First 3 features are the coordinates
    if noise is None:
        # noise = generate_per_residue_noise(x_data)
        noise = torch.randn_like(x_start)
    x_noisy = sampler.q_sample(x_start=x_start,
                               t=t,
                               noise=noise,
                               )
    x_noisy = torch.cat((x_noisy[:,:3], x_data.x[:,3:]), dim=1)
    x_data.x = x_noisy
    predicted_noise = denoise_model(x_data, t)
    noise[:, 3:] = x_start[:, 3:]  # masked coords

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss_copy = F.smooth_l1_loss(noise[:, 3:], predicted_noise[:, 3:])
        loss_denoise = F.smooth_l1_loss(noise[:, :3], predicted_noise[:, :3])
        loss = 0.3 * loss_copy + 0.7 * loss_denoise
    else:
        raise NotImplementedError()

    return loss, loss_denoise