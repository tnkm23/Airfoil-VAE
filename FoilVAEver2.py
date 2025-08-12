"""
Title: Airfoil AutoEncoder
Author: tnkm23
Date created: 2025/08/12
Last modified: 2025/08/12
Description: Convolutional Variational AutoEncoder (VAE) trained on Airfoil shapes.
Accelerator: CPU

VAE
 入力：Airfoil形状データ
 出力：Airfoil形状データの再構成

"""
"""
## Setup
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers


# -----------------------
# ハイパーパラメータ
# -----------------------
DATA_PATH = "airfoils_resampled.npy"
MODEL_DIR = "vae_models"
BATCH_SIZE = 32
EPOCHS = 30
LATENT_DIM = 2         # 潜在次元（調整する）
N_POINTS = 200
N_CHANNELS = 2         # x,y
LEARNING_RATE = 1e-3


"""
## Create a sampling layer
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        """
        reparametrization trick（再パラメータ化トリック）
        ガウシアン分布からサンプリングするためのトリック。
        z_meanとz_log_varを使って、zをサンプリングする。
        サンプリング操作を微分可能にし、VAEの学習を可能にしています。
        潜在空間の分布から「z」を生成。
        """
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


"""
## Build the encoder
"""
encoder_inputs = keras.Input(shape=(N_POINTS, N_CHANNELS), name="encoder_input")
x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(encoder_inputs)
x = layers.MaxPooling1D(pool_size=2, padding="same")(x)   # 200 -> 100
x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
x = layers.MaxPooling1D(pool_size=2, padding="same")(x)   # 100 -> 50
x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


"""
## Build the decoder
"""
latent_inputs = keras.Input(shape=(LATENT_DIM,), name="z_sampling")
x = layers.Dense(50 * 128, activation="relu")(latent_inputs)
x = layers.Reshape((50, 128))(x)  # match encoder downsampled length
x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
x = layers.UpSampling1D(size=2)(x)   # 50 -> 100
x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
x = layers.UpSampling1D(size=2)(x)   # 100 -> 200
x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(x)
decoded_outputs = layers.Conv1D(N_CHANNELS, kernel_size=1, padding="same", activation=None, name="decoder_output")(x)
decoder = keras.Model(latent_inputs, decoded_outputs, name="decoder")
decoder.summary()


"""
## Define the VAE as a `Model` with a custom `train_step`
"""

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """
        1. エンコーダを通してデータを潜在変数に変換
        2. デコーダを通して再構成データを生成
        3. 再構成損失とKL損失（正則化項）を計算
        4. 勾配を計算し、モデルを更新
        z_mean：潜在変数の平均 myu
        z_log_var：潜在変数の対数分散 log(var(z)) log(sigma^2)
        var_z：潜在変数の分散 var(z) sigma^2
        z：潜在変数
     """
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            
            # デコーダを通して再構成データを生成
            reconstruction = self.decoder(z)

            # 再構成損失        
            reconstruction_loss = ops.mean(
                ops.sum(
                    ops.square(data - reconstruction),
                    axis=1  # (batch, 200, 2) → sum over points/channels
                )
            )

            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var)) # KLダイバージェンス
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss # VAEの総損失

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


"""
## Train the VAE

- airfoils_resampled.npyを読み込み。airfoilデータは(1640,200,2)のndarray
- yチャネルのスケール正規化（必要なら）
- VAEモデルを作成・コンパイルし、全データで学習
"""

# Airfoilデータの読み込み
airfoil_data = np.load("airfoils_resampled.npy").astype("float32")  # shape: (1637, 200, 2)

# 必要ならyチャンネルのスケール調整
y_std = np.std(airfoil_data[:,:,1])
if y_std > 0:
    airfoil_data[:,:,1] = airfoil_data[:,:,1] / y_std

# VAEモデルの作成・コンパイル
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

# 学習
vae.fit(airfoil_data, epochs=EPOCHS, batch_size=BATCH_SIZE)

"""
## Display a grid of sampled airfoil
"""

import matplotlib.pyplot as plt

def plot_latent_space_airfoil(vae, n=10, figsize=(15, 8)):
    scale = 1.0
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    fig, axes = plt.subplots(n, n, figsize=figsize)
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            coords_decoded = vae.decoder.predict(z_sample, verbose=0)[0]  # shape: (200, 2)
            ax = axes[i, j]
            ax.plot(coords_decoded[:,0], coords_decoded[:,1], 'b-')
            ax.axis('equal')
            ax.axis('off')
    plt.tight_layout()
    plt.show()


plot_latent_space_airfoil(vae)

"""
## Display how the latent space clusters different airfoil classes
"""


def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the airfoil classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data, verbose=0)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

"""
「潜在空間の可視化（クラスタリングプロット）用に、画像とラベルを準備」
"""
# (x_train, y_train), _ = keras.datasets.mnist.load_data()
# x_train = np.expand_dims(x_train, -1).astype("float32") / 255

# plot_label_clusters(vae, x_train, y_train)
