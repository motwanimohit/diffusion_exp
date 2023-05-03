import torch
import matplotlib.pyplot as plt
from diffusion.beta_scheduler import T, forward_diffusion
from diffusion.dataloader import dataloader, show_tensor_image


image = next(iter(dataloader))[0]
plt.figure(figsize=(15, 15))
plt.axis('off')

num_images = 10
stepsize = int(T/ num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, (idx//stepsize)+ 1)
    image, noise = forward_diffusion(image, t)
    show_tensor_image(image)
    # plt.savefig("diffusion.png")
    print("Making a change")
    print("Making a change")
