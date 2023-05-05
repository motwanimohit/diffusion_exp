
import torch
import os

from diffusion.sample import sample_plot_image
from diffusion.beta_scheduler import forward_diffusion, T
from diffusion.unet import SimpleUnet
from diffusion.dataloader import dataloader

model = SimpleUnet()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
BATCH_SIZE = 64
epochs = 200 # Try more!
training_image_path = '/app/data/training'

os.makedirs(training_image_path, exist_ok=True)

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion(x_0, t)
    noise_pred = model(x_noisy, t)
    return torch.nn.functional.l1_loss(noise, noise_pred)

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch[0], t)
      loss.backward()
      optimizer.step()

      if epoch % 5 == 0 and step == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        # print(os.getcwd())
        final_path = f'{training_image_path}/epoch_{epoch}_step_{step}.png'  
        sample_plot_image(model, path=final_path)

torch.save(model.state_dict(), '/app/data/model.pth')