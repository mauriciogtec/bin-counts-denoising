import numpy as np
from numpy import random
import tensorflow as tf
import matplotlib.pyplot as plt


def generate_toy_data(N: int, nbins: int, mu: float, sigma: float):
    assert mu < sigma**2, "mu < sigma**2"
    r = mu**2 / (sigma**2 - mu)
    p = r / (r + mu)
    theta = np.zeros(nbins, float)
    beta = np.zeros(nbins, int)
    for i in range(5):
        u = np.random.rand()
        pos = np.random.randint(1, nbins - 1)
        theta[pos] = u
        theta[pos + 1] = 0.5 * u
        theta[pos - 1] = 0.5 * u
    beta = np.random.poisson(np.round(N * theta))
    noise = np.random.negative_binomial(r, p, size=nbins)
    y = beta + noise
    return {'obs': y, 'latent': beta, 'noise': noise, 'r': r, 'p': p}


# test
N = 1000
nbins = 100
nbins
mu = 2.0
sigma = 10.0
test = generate_toy_data(N, nbins, mu, sigma)
obs = tf.constant(test["obs"], tf.float32)


# use tensorflow to define a model
# 1. Differce matrix
def generate_diffmat(nbins: int):
    D = np.zeros((nbins - 1, nbins))
    for i in range(nbins - 1):
        D[i, i] = 1
        D[i, i + 1] = -1
    return tf.constant(D, tf.float32)


D = generate_diffmat(nbins)


# 2. Model variables
eta_t = tf.Variable(tf.zeros([nbins], tf.float32))
eta_bg = tf.Variable(tf.constant(0.0))
log_r = tf.Variable(tf.constant(1.0))
var_list = [eta_bg, eta_t, log_r]


@tf.function
def get_loglikelihood():
    #
    r = tf.math.exp(log_r)
    p = 1.0 - tf.math.sigmoid(eta_bg + eta_t)
    #
    logp_negbinom =\
        r * tf.math.log(p + 1e-12) +\
        obs * tf.math.log(1.0 - p + 1e-12) +\
        tf.math.lgamma(r + obs) +\
        - tf.math.lgamma(r) +\
        - tf.math.lgamma(obs + 1)  # not needed for the optim
    #
    return tf.reduce_sum(logp_negbinom)


# 3. Define loss
@tf.function
def get_loss(return_all=False):
    tv = D @ tf.expand_dims(eta_t, 1)
    tv = tf.reduce_sum(tf.math.abs(tv))
    reg = tf.reduce_sum(tf.math.abs((eta_t + 10.0)))
    neglogll = - get_loglikelihood()
    loss = neglogll + tv + reg
    if not return_all:
        return loss
    else:
        return loss, neglogll, reg, tv


# 4. Optimize
opt = tf.optimizers.Adam(learning_rate=0.1)


# get grads
losses = []
negloglls = []
regs = []
tvs = []

mu_hat = []

# initialize logodds
eta_t.assign(tf.math.log(1.0 + obs))

# optimize
for i in range(1000):
    opt.minimize(get_loss, var_list)
    #
    loss, neglogll, reg, tv = get_loss(return_all=True)
    losses.append(loss.numpy())
    negloglls.append(neglogll.numpy())
    regs.append(reg.numpy())
    tvs.append(tv.numpy())
    #
    mu_clean = np.exp(eta_t.numpy()) * np.exp(log_r.numpy())
    mu_hat.append(mu_clean)
    #
    # eta_bg.assign(tf.clip_by_value(eta_bg, -3.0, 3.0))
    # eta_t.assign(tf.clip_by_value(eta_t, -3.0, 3.0))
    #
    0

0