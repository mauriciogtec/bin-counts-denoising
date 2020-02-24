# %%
from neural_denoiser import BinDenoiser
from negbin import RecordGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# %%
maxbins = None
denoiser = BinDenoiser(nblocks=5, ksize=10)
inputs = tf.keras.Input(shape=(maxbins, 1))
outputs = denoiser(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()


# %%
def my_loss(y, yhat):
    err = y - yhat
    err2 = tf.reduce_sum(err**2, axis=1)
    loss = tf.reduce_mean(err2)
    x = tf.expand_dims(yhat, axis=-1)  # batch x nbins x 1
    # x = tf.expand_dims(x, axis=-1)  # batch x nbins x 1 x 1
    tv = tf.image.total_variation(x)  # batch
    reg = 0.001 * tf.reduce_mean(tv)
    return loss + reg


model.compile(optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-3,
                clipnorm=1.0),
              loss=my_loss)


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
lamb = 0.1
losses_av = []
batch = 16
for sim in range(nsims):
    n_bins = np.random.randint(10, 501)
    Xbatch = []
    Ybatch = []
    for _ in range(batch):
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

    if sim % 100 == 0:
        plot_test_case()
        plt.show()

    if sim % 1000 == 0:
        model.save_weights("tmp.h5")

    print(f"sim {sim}/{nsims}, loss: {losses[-1]}, losses_av: {losses_av[-1]}")


# %%

plt.plot(losses)
plt.plot(losses_av, c="red")


# %%
x, y, yhat = plot_test_case()
plt.scatter(yhat, y)
plt.xlabel("prediction")
plt.ylabel("truth")
M = max(max(yhat), max(y))
plt.plot([0, M], [0, M], c="red")


# %%


# %%
