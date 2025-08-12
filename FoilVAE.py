# vae_airfoil_conv1d.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# -----------------------
# ハイパーパラメータ
# -----------------------
DATA_PATH = "airfoils_resampled.npy"
MODEL_DIR = "vae_models"
BATCH_SIZE = 32
EPOCHS = 120
LATENT_DIM = 8         # 潜在次元（調整する）
N_POINTS = 200
N_CHANNELS = 2         # x,y
LEARNING_RATE = 1e-3

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# データ読み込み / 前処理
# -----------------------
data = np.load(DATA_PATH)  # shape: (N, 200, 2)
print("Loaded data:", data.shape)

# optional: check and rescale y-channel if necessary (x is already 0..1)
# Here we scale y to have roughly unit std to stabilize training:
y_std = np.std(data[:,:,1])
if y_std > 0:
    data[:,:,1] = data[:,:,1] / (y_std)

# split
from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(data, test_size=0.12, random_state=42)
print("train:", x_train.shape, "test:", x_test.shape)

# create tf datasets
train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE)

# -----------------------
# VAE コンポーネント定義
# -----------------------
# Sampling (reparameterization)
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

# Encoder
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

# Decoder
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

# -----------------------
# VAE モデル（カスタムトレーニングループを含む）
# -----------------------
class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # beta-VAE 選択可
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]
    
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            recon = self.decoder(z, training=True)
            # Reconstruction loss: mean squared error over all coordinates
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - recon), axis=[1,2]))
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = recon_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }
    
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data, training=False)
        recon = self.decoder(z, training=False)
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - recon), axis=[1,2]))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        total_loss = recon_loss + self.beta * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

vae = VAE(encoder, decoder, beta=1.0)
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
vae.compile(optimizer=optimizer)

# -----------------------
# コールバック
# -----------------------
callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "vae_best.h5"), save_weights_only=True, save_best_only=True, monitor="loss"),
    keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=8),
    keras.callbacks.EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)
]

# -----------------------
# 学習
# -----------------------
history = vae.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, callbacks=callbacks)

# 保存（最後のモデル）
vae.save_weights(os.path.join(MODEL_DIR, "vae_last.h5"))

# -----------------------
# 生成・復元の例
# -----------------------
# 1) 既存データのエンコード→復元
sample = x_test[:6]
z_mean, z_log_var, z = encoder.predict(sample)
recon = decoder.predict(z)

# recon shape = (6, 200, 2)
# 2) 潜在空間からサンプリングして新規生成
n_gen = 12
z_samples = np.random.normal(size=(n_gen, LATENT_DIM))
gen_shapes = decoder.predict(z_samples)  # shape (n_gen, 200, 2)

# 生成結果を元のスケールに戻したい場合は、yのスケールを掛け戻す
gen_shapes[:,:,1] = gen_shapes[:,:,1] * (y_std if y_std>0 else 1.0)

# -----------------------
# dat形式で書き出す関数（XFOIL互換）
# -----------------------
def save_airfoil_dat(coords, filename, name="generated"):
    """
    coords: (N,2) numpy array expected x descending or ascending?
    UIUC dat convention: start at upper surface trailing edge -> around leading edge -> lower surface trailing edge
    Here we assume coords is ordered in that manner or at least is continuous around the foil.
    """
    with open(filename, "w") as f:
        f.write(name + "\n")
        for x,y in coords:
            f.write(f"{x:.6f} {y:.6f}\n")

# Example write
for i in range(len(gen_shapes)):
    save_airfoil_dat(gen_shapes[i], f"generated_{i}.dat", name=f"gen_{i}")

print("Done. Models and some generated .dat files saved.")
