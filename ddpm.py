# https://github.com/dome272/Diffusion-Models-pytorch
import os
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from utils.unet import UNet
from utils.schedules import ddpm_schedules

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, n_T=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__()
        self.n_T = n_T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        for k, v in ddpm_schedules(beta_start, beta_end, n_T).items():
            setattr(self, k, v.to(self.device))

    def forward(self, x):
        ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        eps = torch.randn_like(x).to(self.device)

        # sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        noisy_x = self.sqrtab[ts, None, None, None] * x + self.sqrtmab[ts, None, None, None] * eps

        return noisy_x, eps, ts

    def sample(self, eps_model, n, image_shape):
        logging.info(f"Sampling {n} new images....")

        with torch.no_grad():
            x = torch.randn((n, *image_shape)).to(self.device)

            for i in tqdm(range(self.n_T, 0, -1)):
                t = torch.empty(n).fill_(i).to(self.device)
                eps = eps_model(x, t)
                z = torch.randn((n, *image_shape)).to(self.device) if i > 1 else 0.0

                x = self.oneover_sqrta[i] * (x - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

        x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0
        x = x.type(torch.uint8)

        return x


def train():
    # hyperparams
    epochs = 20
    batch_size = 64
    device = "cuda"
    lr = 1e-4

    # data
    tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = MNIST("./data", train=True, download=False, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    # model
    diffusion = Diffusion(n_T=1000, beta_start=1e-4, beta_end=0.02)
    eps_model = UNet(c_in=1, c_out=1).to(device)
    optimizer = optim.AdamW(eps_model.parameters(), lr=lr)
    mse = nn.MSELoss()
    # logger = SummaryWriter(os.path.join("runs", run_name))

    l = len(dataloader)
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}/{epochs}:")

        pbar = tqdm(dataloader)
        loss_ema = None
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            noisy_images, noise, ts = diffusion.forward(images)
            predicted_noise = eps_model(noisy_images, ts)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_postfix(loss_ema=loss_ema)
            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # sampled_images = diffusion.sample(model, n=images.shape[0])
        # save_images(sampled_images, os.path.join("results", run_name, f"{epoch}.jpg"))

    # save weights
    os.makedirs("model_weights", exist_ok=True)
    torch.save(eps_model.state_dict(), os.path.join("model_weights", "mnist_weights.pt"))


def inference():
    diffusion = Diffusion(n_T=1000, beta_start=1e-4, beta_end=0.02)
    eps_model = UNet(c_in=1, c_out=1)
    eps_model.load_state_dict(torch.load(os.path.join("model_weights", "mnist_weights.pt")))
    eps_model.to("cuda").eval()

    n = 16
    img_shape = (1, 32, 32)
    nrow = 4
    samples = diffusion.sample(eps_model, n, img_shape).detach().cpu()
    plt.imshow(
        make_grid(samples, nrow=nrow).permute(1, 2, 0)
    )
    plt.show()


if __name__ == '__main__':
    # train()
    inference()
