import torch
import torch.nn.functional as F
from tqdm import tqdm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def generate_per_residue_noise(x_data, eps=1e-3):
    x_start = x_data.x.contiguous()
    atoms = x_data.x[:, -4:].sum(dim =1)
    c_n_atoms = torch.where(atoms == 1)[0].to(x_start.device)
    p_atoms = torch.where(atoms == 0)[0].to(x_start.device)
    per_residue_noise = torch.rand((c_n_atoms.shape[0])//4, x_start.shape[1], device=x_start.device) # generate noise for each C4' atom
    per_residue_noise = torch.repeat_interleave(per_residue_noise, 4, dim=0) # repeat it for all atoms in residue (except for P)
    noise = torch.zeros_like(x_start)
    noise[c_n_atoms] = per_residue_noise
    diff = torch.arange(0, len(p_atoms), device=x_start.device)
    relative_c4p = p_atoms - diff # compute the index of each C4' for every P atom
    noise[p_atoms] = noise[c_n_atoms[relative_c4p]] # if there is a P atom, copy the noise from the corresponding C4' atom
    noise = noise + torch.randn_like(x_start, device=x_start.device) * eps

    return noise

class Sampler():
    def __init__(self, timesteps: int, channels: int=3):
        self.timesteps = timesteps
        self.channels = channels
        # define beta schedule
        # self.betas = cosine_beta_schedule(timesteps=timesteps)
        self.betas = linear_beta_schedule(timesteps=timesteps)

        # define alphas 
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        
    @torch.no_grad()
    def p_sample(self, model, x_raw, t, t_index, coord_mask, atoms_mask):
        x = x_raw.x * coord_mask
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x_raw, t)*coord_mask / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            x_raw.x = model_mean * coord_mask + x_raw.x * atoms_mask
            return x_raw.x
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            out = model_mean + torch.sqrt(posterior_variance_t) * noise
            x_raw.x = out * coord_mask + x_raw.x * atoms_mask
            return x_raw.x


    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(self, model, shape, context_mols):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        coord_mask = torch.ones_like(context_mols.x)
        coord_mask[:, 3:] = 0
        atoms_mask = 1 - coord_mask
        noise = torch.rand_like(context_mols.x, device=device)
        # noise = generate_per_residue_noise(context_mols)
        denoised = []
        
        context_mols.x = noise * coord_mask + context_mols.x * atoms_mask
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            context_mols.x = self.p_sample(model, context_mols, torch.full((b,), i, device=device, dtype=torch.long), i, coord_mask, atoms_mask)
            denoised.append(context_mols.clone().cpu())
        return denoised


    @torch.no_grad()
    def sample(self, model, context_mols):
        return self.p_sample_loop(model, shape=context_mols.x.shape, context_mols=context_mols)


    # forward diffusion (using the nice property)
    def q_sample(self,
                 x_start,
                 t,
                 noise=None
                 ):
        if noise is None:
            noise = torch.randn_like(x_start)
        

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)