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
    theta[nbins // 3] = 0.1
    theta[2 * nbins // 3] = 0.8
    beta = random.poisson(np.round(N * theta))
    noise = random.negative_binomial(r, p, size=nbins)
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
beta = tf.Variable(tf.zeros(nbins), tf.float32)
mu = tf.Variable(0.1, tf.float32)
sigma = tf.Variable(0.5, tf.float32)
var_list = [beta, mu, sigma]


# 3. Define loss
@tf.function
def get_loss():
    # r = mu ** 2 / (sigma ** 2 - mu)
    # p = r / (r + mu)
    noise = obs - beta
    # logll = r * tf.math.log(p) + noise * tf.math.log(1.0 - p)\
    #     + tf.math.lgamma(r + noise)\
    #     - tf.math.lgamma(noise + 1.0)\
    #     - tf.math.lgamma(r)
    logll = - tf.math.pow(noise, 2)
    logll = tf.math.reduce_sum(logll)
    tv_penalty = tf.matmul(D, tf.expand_dims(beta, 1))
    tv_penalty = tf.math.pow(tv_penalty, 2)
    tv_penalty = 0.01 * tf.math.reduce_sum(tv_penalty)
    reg_penalty = 0.0  # 0.001 * mu ** 2 + 0.001 * sigma ** 2
    loss = - logll + tv_penalty + reg_penalty
    return loss


# 4. Optimize
opt = tf.optimizers.Adam(learning_rate=1.0)


# get grads
var_list = [beta]
losses = []
losses.append(get_loss().numpy())
for i in range(1000):
    opt.minimize(get_loss, var_list)
    losses.append(get_loss().numpy())
    # beta.assign(tf.clip_by_value(beta, 1e-6, obs))
    # mu.assign(tf.maximum(mu, 1e-6))
    # sigma.assign(tf.maximum(sigma, tf.math.sqrt(mu) + 1e-3))
    0

# with tf.GradientTape() as g:
#     loss_ = get_loss()
# grad = g.gradient(loss_, var_list)
# opt.apply_gradients(zip(grad, var_list))
# 
0