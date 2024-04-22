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
    if noise is None:
        noise = torch.randn_like(x_data)

    x_noisy = sampler.q_sample(x_start=x_data,
                               t=t,
                               noise=noise,
                               )
    
    predicted_noise = denoise_model(x_noisy, t) # TODO: Model aware of the timestep t?

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss