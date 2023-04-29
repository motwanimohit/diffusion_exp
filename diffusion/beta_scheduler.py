import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int , start: float=0.0001, end : float= 0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals: torch.tensor, t: torch.tensor, x_shape: tuple):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion(x_0, t, device = 'cpu'):
    """
    Takes an image and a timestep t as input and returns the noisy version of the image
    """
    
    #noise from normal distribution like x_0
    noise = torch.randn_like(x_0, device=device)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    #mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise

T = 200
betas = linear_beta_schedule(timesteps = T)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

#pad the last dimention only on one side (1, 0) with value 1.0
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)


sqrt_recip_alphas = torch.sqrt(1.0/ alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

