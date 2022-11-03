# https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import Progbar
from tensorflow_addons.layers import GroupNormalization, GELU
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class ResidualConvBlock(layers.Layer):
    def __init__(self, in_channels, out_channels: int, is_res: bool = False):
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = models.Sequential([
            layers.Conv2D(out_channels, 3, 1, padding="same"),
            layers.BatchNormalization(),
            GELU(),
        ])
        self.conv2 = models.Sequential([
            layers.Conv2D(out_channels, 3, 1, padding="same"),
            layers.BatchNormalization(),
            GELU(),
        ])

    def call(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()

        self.conv = ResidualConvBlock(in_channels, out_channels)
        self.pool = layers.MaxPooling2D(2)

    def call(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class UnetUp(layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        self.conv_transpose = layers.Conv2DTranspose(out_channels, 2, 2, padding="same")
        self.conv1 = ResidualConvBlock(in_channels, out_channels)
        self.conv2 = ResidualConvBlock(in_channels, out_channels)
        self.concate = layers.Concatenate()

    def call(self, x, skip):
        x = self.concate([x, skip])
        x = self.conv_transpose(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class EmbedFC(layers.Layer):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        self.fc1 = layers.Dense(emb_dim)
        self.gelu = GELU()
        self.fc2 = layers.Dense(emb_dim)

    def call(self, x):
        x = tf.reshape(x, (-1, self.input_dim))
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x


class ContextUnet(models.Model):
    def __init__(self, in_channels, n_feat=256, n_classes=10):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = models.Sequential([
            layers.AveragePooling2D(7),
            GELU()
        ])

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1 * n_feat)

        self.up0 = models.Sequential([
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            layers.Conv2DTranspose(2 * n_feat, 7, 7),  # otherwise just have 2*n_feat
            GroupNormalization(8),
            layers.ReLU(),
        ])

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = models.Sequential([
            layers.Conv2D(n_feat, 3, 1, padding="same"),
            GroupNormalization(8),
            layers.ReLU(),
            layers.Conv2D(self.in_channels, 3, 1, padding="same"),
        ])
        self.concate = layers.Concatenate()

    def call(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = tf.one_hot(c, self.n_classes, dtype=tf.float32)

        # mask out context if context_mask == 1
        context_mask = tf.expand_dims(context_mask, -1)
        context_mask = tf.tile(context_mask, (1, self.n_classes))
        context_mask = -1 * (1 - context_mask)  # need to flip 0 <-> 1
        context_mask = tf.cast(context_mask, tf.float32)
        c = c * context_mask

        # embed context, time step
        cemb1 = self.contextembed1(c)
        temb1 = self.timeembed1(t)
        cemb1 = tf.reshape(cemb1, (-1, 1, 1, self.n_feat * 2))
        temb1 = tf.reshape(temb1, (-1, 1, 1, self.n_feat * 2))

        cemb2 = self.contextembed2(c)
        temb2 = self.timeembed2(t)
        cemb2 = tf.reshape(cemb2, (-1, 1, 1, self.n_feat))
        temb2 = tf.reshape(temb2, (-1, 1, 1, self.n_feat))

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(self.concate([up3, x]))

        return out


def ddpm_schedules(beta1, beta2, T):
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


class DDPM(models.Model):
    def __init__(self, eps_model, betas, n_T, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.eps_model = eps_model
        self.params = ddpm_schedules(betas[0], betas[1], n_T)
        self.n_T = n_T
        self.drop_prob = drop_prob

    def forward(self, x, c):
        _ts = np.random.randint(1, self.n_T, (x.shape[0],))  # t ~ Uniform(0, n_T)
        noise = np.random.randn(*x.shape)  # eps ~ N(0, 1)

        x_t = (
                np.reshape(self.params["sqrtab"][_ts], (-1, 1, 1, 1)) * x
                + np.reshape(self.params["sqrtmab"][_ts], (-1, 1, 1, 1)) * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        context_mask = tfp.distributions.Bernoulli(probs=self.drop_prob).sample(c.shape)

        # return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))
        return x_t, noise, _ts / self.n_T, context_mask

    def call(self, x, c, t, context_mask):
        return self.eps_model(x, c, t, context_mask)

    def sample(self, n_sample, size, c, guide_w=0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        assert len(c) == n_sample

        x_i = np.random.randn(n_sample, *size)  # x_T ~ N(0, 1)
        c_i = c

        # don't drop context at test time
        context_mask = np.zeros_like(c_i)

        # double the batch
        c_i = np.tile(c_i, 2)
        context_mask = np.tile(context_mask, 2)
        context_mask[n_sample:] = 1.  # makes second half of batch context free

        x_i_store = [
            np.clip((x_i + 1) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        ]

        for i in tqdm(range(self.n_T, 0, -1)):
            t_is = i / self.n_T
            t_is = np.tile(t_is, (n_sample, 1, 1, 1))

            # double batch
            x_i = np.tile(x_i, (2, 1, 1, 1))
            t_is = np.tile(t_is, (2, 1, 1, 1))

            z = np.random.randn(n_sample, *size) if i > 1 else 0.0

            # split predictions and compute weighting
            eps = self.eps_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = self.params["oneover_sqrta"][i] * (x_i - eps * self.params["mab_over_sqrtmab"][i]) + \
                  self.params["sqrt_beta_t"][i] * z

            if i % 20 == 0 or i == self.n_T or i < 8:
                # scale to [0, 255]
                x_i_store.append(
                    np.clip((x_i.numpy() + 1) / 2.0 * 255.0, 0, 255).astype(np.uint8)
                )

        return x_i_store[-1], x_i_store


class DDIM(DDPM):
    def __init__(self, eps_model, betas, n_T, sigma_t=0.0, drop_prob=0.1):
        super().__init__(eps_model, betas, n_T, drop_prob)
        self.sigma_t = sigma_t

    def sample(self, n_sample, size, c, inference_steps=100, guide_w=0.0):
        assert len(c) == n_sample

        x_i = np.random.randn(n_sample, *size)  # x_T ~ N(0, 1)
        c_i = c

        # don't drop context at test time
        context_mask = np.zeros_like(c_i)

        # double the batch
        c_i = np.tile(c_i, 2)
        context_mask = np.tile(context_mask, 2)
        context_mask[n_sample:] = 1.  # makes second half of batch context free

        x_i_store = [np.clip((x_i + 1) / 2.0 * 255.0, 0, 255).astype(np.uint8)]
        inter_step_size = self.n_T // inference_steps
        for i in tqdm(range(self.n_T, 0, -inter_step_size)):
            t_is = i / self.n_T
            t_is = np.tile(t_is, (n_sample, 1, 1, 1))

            # double batch
            x_i = np.tile(x_i, (2, 1, 1, 1))
            t_is = np.tile(t_is, (2, 1, 1, 1))

            # split predictions and compute weighting
            eps = self.eps_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]

            # https://miro.medium.com/max/720/1*HnweiKivDAE4JdpTZmD6BA.png
            x_i = (x_i - self.params["sqrtmab"][i] * eps) / self.params["sqrtab"][i]
            x_i *= np.sqrt(self.params["alpha_t"][i - 1])  # sqrt(alpha_t-1)

            x_i += np.sqrt(1 - self.params["alpha_t"][i - 1] - np.square(self.sigma_t)) * eps

            z = np.random.randn(n_sample, *size) if i > 1 else 0.0
            x_i += self.sigma_t * z  # sigma_t * eps_t

            # scale to [0, 255]
            x_i_store.append(
                np.clip((x_i.numpy() + 1) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            )

        return x_i_store[-1], x_i_store


def train_mnist():
    # data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    img_data = np.concatenate([x_train[..., np.newaxis], x_test[..., np.newaxis]], axis=0)
    img_data = img_data / 255.0 * 2.0 - 1.0  # scale images from [0,255] to [-1, 1]
    labels = np.concatenate([y_train, y_test], axis=0)

    # params
    n_epoch = 20
    batch_size = 64
    n_T = 500
    n_classes = 10
    n_feat = 128  # 128 ok, 256 better (but slower)
    lr = 1e-4

    # model
    ddpm = DDPM(eps_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes),
                betas=(1e-4, 0.02), n_T=n_T, drop_prob=0.1)
    optim = tf.keras.optimizers.Adam(lr)

    # training
    n_img = img_data.shape[0]
    progress_bar = Progbar(target=np.ceil(n_img / batch_size), unit_name='step')
    for i in range(n_epoch):
        rnd_idx = list(range(n_img))
        np.random.shuffle(rnd_idx)

        losses = []
        step = 0
        for batch_idx in range(0, n_img, batch_size):
            # data batch
            img_batch = img_data[rnd_idx[batch_idx: batch_idx + batch_size]]
            label_batch = labels[rnd_idx[batch_idx: batch_idx + batch_size]]

            noisy_img, noise, ts, mask = ddpm.forward(img_batch, label_batch)
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(
                    noise - ddpm(noisy_img, label_batch, ts, mask)
                ))
            grads = tape.gradient(loss, ddpm.trainable_weights)
            optim.apply_gradients(
                [(grad, var) for (grad, var) in zip(grads, ddpm.trainable_weights) if grad is not None])

            losses.append(loss.numpy())
            progress_bar.update(step)
            step += 1

        print(f"    epoch: {i: >3}: {np.mean(losses):.4f}")

    ddpm.save_weights("conditional_model/mnist_model/")


def combine_samples(x_gen, n_sample, n_row, n_col):
    assert n_row * n_col == n_sample

    h = x_gen.shape[1]
    w = x_gen.shape[2]
    c = x_gen.shape[3]
    merged_img = np.empty((h * n_row, w * n_col, c))

    for row in range(n_row):
        for col in range(n_col):
            img_idx = row * n_col + col
            merged_img[row * h: row * h + h, col * w: col * w + w] = x_gen[img_idx]

    return merged_img


def save_gif(x_list, path="", interval=200):
    # x_list: list of samples
    imgs = []

    print("Saving gif")
    for x in tqdm(x_list):
        x = combine_samples(x, 40, 4, 10).reshape(28 * 4, 28 * 10).astype(np.uint8)
        imgs.append(Image.fromarray(x))

    imgs[0].save(fp=path, format='GIF', append_images=imgs[1:], save_all=True, duration=interval, loop=0)


def inference_ddpm():
    # model
    n_T = 500
    n_classes = 10
    n_feat = 128
    ddpm = DDPM(eps_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes),
                betas=(1e-4, 0.02), n_T=n_T, drop_prob=0.1)
    ddpm(np.zeros((1, 28, 28, 1)), np.zeros(1), np.zeros(1), np.zeros(1))
    ddpm.load_weights("conditional_model/mnist_model/")

    # sampling
    n_samples = 4 * n_classes
    c = np.tile(np.arange(n_classes), 4)
    w = 1.0  # [0.0, 0.5, 2.0] strength of generative guidance
    x, x_store = ddpm.sample(n_samples, (28, 28, 1), c=c, guide_w=w)

    plt.imshow(
        combine_samples(x, n_samples, 4, 10), cmap="gray"
    )
    plt.show()
    save_gif(x_store, "../samples/conditional_ddpm.gif")


def inference_ddim():
    # model
    n_T = 500
    n_classes = 10
    n_feat = 128
    inference_steps = 2
    ddpm = DDIM(eps_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes),
                betas=(1e-4, 0.02), n_T=n_T, drop_prob=0.1)
    ddpm(np.zeros((1, 28, 28, 1)), np.zeros(1), np.zeros(1), np.zeros(1))
    ddpm.load_weights("conditional_model/mnist_model/")

    # sampling
    n_samples = 4 * n_classes
    c = np.tile(np.arange(n_classes), 4)
    w = 0.5  # [0.0, 0.5, 2.0] strength of generative guidance
    x, x_store = ddpm.sample(n_samples, (28, 28, 1), c=c, inference_steps=inference_steps, guide_w=w)

    plt.imshow(
        combine_samples(x, n_samples, 4, 10), cmap="gray"
    )
    plt.show()
    save_gif(x_store, "../samples/conditional_ddim.gif", interval=500)



if __name__ == "__main__":
    # train_mnist()
    # inference_ddpm()
    inference_ddim()
