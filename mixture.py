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
log_theta = tf.Variable(tf.zeros([nbins], tf.float32))
Z = tf.constant(np.random.choice([0., 1.], size=nbins), tf.float32)
logodds_alpha = tf.Variable(tf.zeros([nbins], tf.float32))
# logodds_p = tf.Variable(0.0, tf.float32)
log_r = tf.Variable(0.0, tf.float32)
var_list = [log_theta, logodds_alpha, log_r]


# 3. hyperperams
a0 = 0.005  # a_t ~ Beta(a0, b0)
b0 = 0.005
lam0 = (N / nbins)  # theta_t ~ Exp(1 / lam0)
# p, r degenerate prior


# @tf.function
def assign_latent():
    r = tf.math.exp(log_r)
    # p = tf.math.sigmoid(logodds_p)
    theta = tf.math.exp(log_theta)
    alpha = 0.999 * tf.math.sigmoid(logodds_alpha) + 0.0005
    #
    # logp_negbinom =\
    #     r * tf.math.log(p + 1e-12) +\
    #     obs * tf.math.log(1.0 - p + 1e-12) +\
    #     tf.math.lgamma(r + obs) +\
    #     - tf.math.lgamma(r) +\
    #     - tf.math.lgamma(obs + 1)
    logp_negbinom = - obs / r - log_r
    #
    logp_exp = - obs / theta - log_theta
    #
    gamma0 = tf.math.log(alpha + 1e-12) + logp_negbinom
    gamma1 = tf.math.log(1.0 - alpha + 1e-12) + logp_exp
    logits = tf.stack([gamma0, gamma1], 1)
    probs = tf.math.sigmoid(logits)
    prob = probs[:, 0]
    ans = tf.random.categorical(logits, 1)
    ans = tf.squeeze(tf.cast(ans, tf.float32))
    return ans, prob, logp_negbinom, logp_exp


# 3. Define loss
# @tf.function
def get_loss():
    theta = tf.math.exp(log_theta)
    r = tf.math.exp(log_r)
    #
    _, prob, logp_negbinom, logp_exp = assign_latent()
    #
    alpha = tf.math.sigmoid(logodds_alpha)
    logp_alpha =\
        (Z + a0 - 1.0) * tf.math.log(alpha) +\
        (1.0 - Z + b0 - 1.0) * tf.math.log(1.0 - alpha)
    #
    logp =\
        Z * logp_negbinom +\
        (1.0 - Z) * logp_exp +\
        logp_alpha
    logp = tf.math.reduce_sum(logp)
    #
    tv_theta = D @ tf.expand_dims(log_theta, 1)
    tv_theta = tf.reduce_sum(tv_theta)
    tv_alpha = D @ tf.expand_dims(logodds_alpha, 1)
    tv_alpha = tf.reduce_sum(tv_alpha)
    reg_loss =\
        0.01 * tf.reduce_sum(theta**2) +\
        0.01 * tf.reduce_sum(r**2) +\
        0.1 * tv_theta +\
        0.1 * tv_alpha  # +\
        # 1.0 / (tf.math.exp(log_r) + 0.1) ** 2 +\
        # 0.01 * logodds_p ** 2 +\
        # theta / lam0
    #
    loss = -logp + reg_loss
    return loss


# 4. Optimize
opt = tf.optimizers.Adam(learning_rate=0.1)


# get grads
losses = []
losses.append(get_loss().numpy())
for i in range(1000):
    # expectation step
    opt.minimize(get_loss, var_list)
    logodds_alpha.assign(tf.clip_by_value(logodds_alpha, -10.0, 10.0))
    # maximizations step
    Z, _, _, _ = assign_latent()
    #
    r = np.exp(log_r.numpy())
    # p = tf.math.sigmoid(logodds_p).numpy()
    # mu = r * p / (1.0 - p)
    # sigma = np.sqrt(mu + mu**2 / r)
    #
    losses.append(get_loss().numpy())
    0

# with tf.GradientTape() as g:
#     loss_ = get_loss()
# grad = g.gradient(loss_, var_list)
# opt.apply_gradients(zip(grad, var_list))
# 
0