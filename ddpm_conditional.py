# https://github.com/dome272/Diffusion-Models-pytorch
import os
import copy
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
from utils.unet import ConditionalUNet, EMA
from utils.schedules import ddpm_schedules

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, n_T=1000, beta_start=1e-4, beta_end=0.02, drop_prob=0.1, device="cuda"):
        super().__init__()
        self.n_T = n_T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.drop_prob = drop_prob
        self.device = device

        for k, v in ddpm_schedules(beta_start, beta_end, n_T).items():
            setattr(self, k, v.to(self.device))

    def forward(self, x):
        ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        eps = torch.randn_like(x)
        # sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        noisy_x = self.sqrtab[ts, None, None, None] * x + self.sqrtmab[ts, None, None, None] * eps

        mask = torch.bernoulli(torch.empty(x.shape[0]).fill_(1.0 - self.drop_prob)).to(self.device)

        return noisy_x, eps, ts, mask

    def sample(self, eps_model, n, image_shape, labels, guide_w=2.0):
        logging.info(f"Sampling {n} new images....")

        labels = labels.repeat(2).to(self.device)
        labels[n:] = 0.

        with torch.no_grad():
            x = torch.randn((n, *image_shape)).to(self.device)
            mask = torch.ones(n * 2).to(self.device)

            for i in tqdm(range(self.n_T, 0, -1)):
                x = x.repeat(2, 1, 1, 1)
                t = torch.empty(n * 2).fill_(i).to(self.device)

                eps = eps_model(x, t, labels, mask)
                eps1 = eps[:n]
                eps2 = eps[n:]
                eps = (1 + guide_w) * eps1 - guide_w * eps2
                x = x[:n]

                z = torch.randn((n, *image_shape)).to(self.device) if i > 1 else 0.0

                x = self.oneover_sqrta[i] * (x - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

        x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0
        x = x.type(torch.uint8)

        return x


def train():
    # hyperparams
    epochs = 50
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
    num_classes = 10
    diffusion = Diffusion(n_T=1000, beta_start=1e-4, beta_end=0.02)
    eps_model = ConditionalUNet(c_in=1, c_out=1, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(eps_model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # ema model
    ema = EMA(0.995)
    ema_model = copy.deepcopy(eps_model).eval().requires_grad_(False)

    l = len(dataloader)
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}/{epochs}:")

        pbar = tqdm(dataloader)
        loss_ema = None
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            noisy_images, noise, ts, mask = diffusion.forward(images)
            predicted_noise = eps_model(noisy_images, ts, labels, mask)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, eps_model)

            # logging
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_postfix(loss_ema=loss_ema)

    # save weights
    os.makedirs("model_weights", exist_ok=True)
    torch.save(eps_model.state_dict(), os.path.join("model_weights", "conditional_mnist_weights.pt"))
    torch.save(eps_model.state_dict(), os.path.join("model_weights", "ema_conditional_mnist_weights.pt"))


def inference():
    num_classes = 10
    diffusion = Diffusion(n_T=1000, beta_start=1e-4, beta_end=0.02)
    eps_model = ConditionalUNet(c_in=1, c_out=1, num_classes=num_classes)
    eps_model.load_state_dict(torch.load(os.path.join("model_weights", "ema_conditional_mnist_weights.pt")))
    eps_model.to("cuda").eval()

    n = 40
    img_shape = (1, 32, 32)
    labels = torch.arange(10).repeat(4)
    nrow = 10
    w = 2.0
    samples = diffusion.sample(eps_model, n, img_shape, labels, guide_w=w).detach().cpu()
    plt.imshow(
        make_grid(samples, nrow=nrow).permute(1, 2, 0)
    )
    plt.show()


if __name__ == '__main__':
    # train()
    inference()
