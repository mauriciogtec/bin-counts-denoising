`# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from neural_denoiser import BinDenoiser
from negbin import RecordGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# %%
maxbins = None
bsize = 16
denoiser = BinDenoiser(nblocks=7, ksize=9, filters=32)
inputs = tf.keras.Input(shape=(maxbins, 1), batch_size=bsize)
outputs = denoiser(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()


# %%
def lasso(x):
    nbins = tf.cast(tf.shape(x), tf.float32)[1]
    x = tf.reduce_sum(tf.math.abs(x), 1) * tf.math.sqrt(nbins)
    x = tf.reduce_mean(x)
    return 0.00005 * x


def tv(x):  
    x = tf.image.total_variation(x)
    x = tf.reduce_mean(x)
    return 0.00005 * x


def my_loss(y, yhat):
    nbins = tf.cast(tf.shape(y), tf.float32)[1]
    loss = tf.reduce_mean((y - yhat)**2) * nbins
    return loss + tv(yhat) + lasso(yhat)
    

model.compile(optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-3,
                clipnorm=1.0),
              loss=my_loss)


# %%
model.load_weights("tmp.h5")


# %%
def plot_test_case():
    data = generator.generate()
    h = np.arange(data['n_bins'])
    x = np.array(data['counts'], dtype=float)
    x /= x.sum()
    y = data['signal_normalized']
    xinput = np.expand_dims(x, -1)
    xinput = np.expand_dims(xinput, 0)
    yhat = np.squeeze(model.predict(xinput))
    plt.bar(h, x, width=1)
    plt.plot(h, yhat, c="red")
    plt.plot(h, y, c="blue")
    return x, y, yhat


# %%
generator = RecordGenerator()
nsims = 50000
X = []
Y = []
losses = []
losses_av = []
lamb = 0.01
losses_av = []

for sim in range(nsims):
    n_bins = np.random.randint(10, 501)
    Xbatch = []
    Ybatch = []
    for _ in range(bsize):
        data = generator.generate(n_bins=n_bins)
        x = np.array(data['counts'], dtype=float)
        x /= x.sum()
        Xbatch.append(x)
        y = data['signal_normalized']
        Ybatch.append(y)
    Xbatch = np.expand_dims(np.stack(Xbatch, 0), -1)
    Ybatch = np.expand_dims(np.stack(Ybatch, 0), -1)
    loss = model.train_on_batch(Xbatch, Ybatch)
    losses.append(float(loss))
    if sim == 0:
        losses_av.append(float(loss))
    else:
        L = lamb * losses[-1] + (1.0 - lamb) * losses_av[-1]
        losses_av.append(L)

    if (sim + 1) % 100 == 0:
        yhat = tf.constant(model.predict(Xbatch))
        tv_ = tv(yhat)
        lasso_ = lasso(yhat)
        print(f"sim {sim + 1}/{nsims}, loss: {losses[-1]:.4f}, ", end="")
        print(f"losses_av: {losses_av[-1]:.4f}, tv: {tv_:.4f}, reg: {lasso_:.4f}")

    if sim == 0 or (sim + 1) % 100 == 0:
        plot_test_case()
        plt.show()
        model.save_weights("tmp.h5")



# %%

plt.plot(losses)
plt.plot(losses_av, c="red")


# %%
plt.scatter(yhat, y)


# %%
x, y, yhat = plot_test_case()
plt.xlabel("prediction")
plt.ylabel("truth")
M = max(max(yhat), max(y))
plt.plot([0, M], [0, M], c="red")


# %%



# %%



