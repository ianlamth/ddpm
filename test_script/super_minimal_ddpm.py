# https://github.com/cloneofsimo/minDiffusion/blob/master/superminddpm.py
from typing import Dict, Tuple
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.utils import Progbar
import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def ddpm_schedules(beta1: float, beta2: float, T: int):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * np.arange(0, T + 1, dtype=np.float32) / T + beta1
    sqrt_beta_t = np.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = np.log(alpha_t)
    alphabar_t = np.exp(np.cumsum(log_alpha_t))

    sqrtab = np.sqrt(alphabar_t)
    oneover_sqrta = 1 / np.sqrt(alpha_t)

    sqrtmab = np.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


blk = lambda ic, oc: models.Sequential([
    layers.Conv2D(oc, 7, padding="same"),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
])
# class blk(layers.Layer):
#     def __init__(self, ic, oc):
#         super(blk, self).__init__()
#         self.conv = layers.Conv2D(oc, 7, padding="same")
#         self.norm = layers.BatchNormalization(oc)
#         self.relu = layers.LeakyReLU()


class DummyEpsModel(models.Model):
    def __init__(self, n_channel):
        super(DummyEpsModel, self).__init__()
        self.conv = models.Sequential([
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            layers.Conv2D(n_channel, 3, padding="same"),
        ])

    def call(self, x, t):
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)


class DDPM(models.Model):
    def __init__(self, eps_model, betas: Tuple[float, float], n_T: int):
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        # for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
        #     self.register_buffer(k, v)
        self.params = ddpm_schedules(betas[0], betas[1], n_T)
        self.n_T = n_T

    def forward(self, x):
        _ts = np.random.randint(1, self.n_T, (x.shape[0],))  # t ~ Uniform(0, n_T)

        # target
        eps = np.random.randn(*x.shape)  # eps ~ N(0, 1)

        x_t = (
            np.reshape(self.params["sqrtab"][_ts], (-1, 1, 1, 1)) * x
            + np.reshape(self.params["sqrtmab"][_ts], (-1, 1, 1, 1)) * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))
        return x_t, eps

    def call(self, x, t):
        return self.eps_model(x, t)

    def sample(self, n_sample: int, size):

        x_i = np.random.randn(n_sample, *size)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = np.random.randn(n_sample, *size) if i > 1 else 0.0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = (
                self.params["oneover_sqrta"][i] * (x_i - eps * self.params["mab_over_sqrtmab"][i])
                + self.params["sqrt_beta_t"][i] * z
            )

        return x_i


def train_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    dataset = np.concatenate([x_train[..., np.newaxis], x_test[..., np.newaxis]], axis=0)
    dataset = dataset / 255.0
    n_img = dataset.shape[0]

    ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    optim = tf.keras.optimizers.Adam(1e-4)

    # train
    batch_size = 128
    n_epoch = 10
    progress_bar = Progbar(target=np.ceil(n_img / batch_size), unit_name='step')
    for i in range(n_epoch):
        rnd_idx = list(range(n_img))
        np.random.shuffle(rnd_idx)

        losses = []
        step = 0
        for batch_idx in range(0, n_img, batch_size):
            img_batch = dataset[rnd_idx[batch_idx: batch_idx + batch_size]]

            noisy_img, noise = ddpm.forward(img_batch)
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(
                    noise - ddpm(noisy_img, None)
                ))
            grads = tape.gradient(loss, ddpm.trainable_weights)
            optim.apply_gradients([(grad, var) for (grad, var) in zip(grads, ddpm.trainable_variables) if grad is not None])

            losses.append(loss.numpy())
            progress_bar.update(step)
            step += 1
        # ddpm.eval()
        # with torch.no_grad():
        #     xh = ddpm.sample(16, (1, 28, 28), device)
        #     grid = make_grid(xh, nrow=4)
        #     save_image(grid, f"./contents/ddpm_sample_{i}.png")
        #
        #     # save model
        #     torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")

        print(f"{i: >3}: {np.mean(losses):.3f}")

    ddpm.save_weights("minimal_model/")


def inference():
    ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    ddpm(np.zeros((1, 28, 28, 1)), None)
    ddpm.load_weights("minimal_model/")

    x = ddpm.sample(16, (28,28,1)).numpy()
    x = x.reshape((16, 28, 28))
    for row in range(4):
        for col in range(4):
            idx = row * 4 + col
            plt.subplot(4, 4, idx + 1)
            plt.imshow(x[idx], cmap="gray")
            plt.axis("off")
    plt.tight_layout()
    plt.show()

# train_mnist()
inference()
