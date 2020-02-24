import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm




def generate_toy_data(N: int, nbins: int, mu: float, r: float):
    x = np.zeros(nbins, dtype=int)
    z1 = np.random.normal(size=N//3, loc=2*nbins//3, scale=nbins//150)
    for z in zip(z1):
        x[int(np.round(z))] += 1
    z2 = np.random.normal(size=2 * N//3, loc=nbins//3, scale=nbins//150)
    for z in zip(z2):
        x[int(np.round(z))] += 1
    p = r / (r + mu)
    eta = np.random.negative_binomial(r, p, size=nbins)
    y = x + eta
    return {'obs': y, 'latent': x, 'noise': eta}


def get_pdfs(N: int, nbins: int):
    x = np.arange(nbins)
    f1 = norm.pdf(x, loc=1*nbins//3, scale=nbins//150)
    f2 = norm.pdf(x, loc=2*nbins//3, scale=nbins//150)
    f = (2.0 / 3.0) * f1 + (1.0 / 3.0) * f2
    return f


N = 300
nbins = 200
toy = generate_toy_data(N, nbins, 3.0, 1.5)

# plt.bar(range(100), toy['obs'])
# plt.show()
# plt.close()


def generate_diffmat(nbins: int):
    D = np.zeros((nbins - 1, nbins))
    for i in range(nbins - 1):
        D[i, i] = -1
        D[i, i + 1] = +1
    return tf.constant(D, tf.float32)


def generate_k_diffmat(nbins: int, k: int):
    D = tf.eye(nbins)
    for i in range(k + 1):
        D = generate_diffmat(nbins - i) @ D
    return D


D0 = generate_k_diffmat(nbins, 0)
D1 = generate_k_diffmat(nbins, 1)
D2 = generate_k_diffmat(nbins, 2)
D3 = generate_k_diffmat(nbins, 3)


# 2. Model variables
n = tf.constant(N, tf.float32)
obs = tf.constant(toy['obs'], tf.float32)
odds = tf.Variable(tf.zeros(nbins, tf.float32))
b = tf.Variable(1.0, tf.float32)
q = tf.Variable(1.0, tf.float32)


@tf.function
def get_loss(return_all=False):
    #
    lam = tf.math.softmax(odds)
    mu = tf.math.exp(-b) + n * lam
    r = tf.math.exp(-q)
    p = r / (r + mu)
    #
    logp_negbinom =\
        r * tf.math.log(p + 1e-12) +\
        obs * tf.math.log(1.0 - p + 1e-12) +\
        tf.math.lgamma(r + obs) - tf.math.lgamma(r) 
    neglogll = - tf.reduce_sum(logp_negbinom) - 0.01 * b**2
    #
    tv = 10.0 * tf.reduce_sum(tf.math.abs(D2 @ tf.expand_dims(odds, 1)))  \
        + 10.0 * tf.reduce_sum(tf.math.abs(D1 @ tf.expand_dims(odds, 1)))  \
        + 0.1 * tf.reduce_sum(tf.math.abs(D0 @ tf.expand_dims(odds, 1)))
    reg = 1.0 * tf.reduce_sum(tf.math.abs(tf.abs(odds) - 10.0))
    loss = neglogll + tv + reg
    return loss, neglogll, tv, reg if return_all else loss


def deg_freedom(x):
    tol = 1e-3
    D2 = generate_k_diffmat(len(x), 2).numpy()
    diffs = np.squeeze(D2 @ np.expand_dims(np.array(x), 1))
    dfs = 1
    current = diffs[0]
    for i in range(1, len(diffs)):
        if np.abs(current - diffs[i]) > tol:
            dfs += 1
            current = diffs[0]
    return dfs


def loglikelihood(odds, b, q):
    lam = tf.math.softmax(np.array(odds)).numpy()
    mu = tf.math.exp(-b) + n * lam
    r = tf.math.exp(-q)
    p = r / (r + mu)
    #
    logp_negbinom =\
        r * tf.math.log(p + 1e-12) +\
        obs * tf.math.log(1.0 - p + 1e-12) +\
        tf.math.lgamma(r + obs) +\
        - tf.math.lgamma(r) +\
        - tf.math.lgamma(obs + 1.0)  # not needed for the optim
    logll = tf.reduce_sum(logp_negbinom)
    return logll.numpy().item()


# Optimize
opt = tf.optimizers.Adam(learning_rate=0.5)
var_list = [odds, b, q]

losses = []
negloglls = []
regs = []
tvs = []
probs = []


# optimize
nsteps = 5000
for i in range(nsteps):
    opt.minimize(get_loss, var_list)
    #
    loss, neglogll, reg, tv = get_loss(return_all=True)
    losses.append(loss.numpy())
    negloglls.append(neglogll.numpy())
    regs.append(reg.numpy())
    tvs.append(tv.numpy())
    probs.append(tf.math.softmax(odds).numpy())
    #
    0

plt.plot(losses)
plt.show()

plt.plot(range(nbins), probs[-1] * N, c="red", label="fit")
plt.plot(range(nbins), N * get_pdfs(N, nbins), c="black", label="true")
plt.bar(range(nbins), toy['obs'], alpha=0.8)
plt.legend()
plt.show()

0
