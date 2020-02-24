import numpy as np
from scipy.stats import norm
# import matplotlib.pyplot as plt


class RecordGenerator():
    def __init__(self,
                 seed=None, n_bins=(10, 500), n_obs=(1000, 100000),
                 noise_ratio=(0.001, 0.75), noise_dispersion=(0.05, 2.0),
                 n_comps=(1, 5), alpha_comps=(0.25, 4.0),
                 rounding=[1, 2, 5, 10],
                 sigmas=(0.001, 25.0), signal_dispersion=(0.00001, 0.4)):
        self.rng = np.random.RandomState(seed)
        self.n_bins = n_bins
        self.n_obs = n_obs
        self.noise_ratio = noise_ratio
        self.noise_dispersion = noise_dispersion
        self.n_points = n_obs
        self.n_comps = n_comps
        self.alpha_comps = alpha_comps
        self.sigmas = sigmas
        self.signal_dispersion = signal_dispersion
        self.rounding = rounding

    def generate(self, n_bins=None):
        if n_bins is None:
            n_bins = self.rng.randint(self.n_bins[0], self.n_bins[1] + 1)
        n_obs = self.rng.randint(self.n_obs[0], self.n_obs[1] + 1)

        # 1. sample the noise component
        noise_ratio = self.rng.uniform(*self.noise_ratio)
        mu_noise = np.ceil(noise_ratio * n_obs / n_bins)
        noise_dispersion = self.rng.uniform(*self.noise_dispersion)
        r = noise_dispersion
        p = r / (r + mu_noise)
        noise = self.rng.negative_binomial(r, p, n_bins)

        # 2. sample the true signal component
        # -- 2.a sample component weights
        alpha = self.rng.uniform(*self.alpha_comps)
        n_comps = self.rng.randint(self.n_comps[0], self.n_comps[1] + 1)
        wts = self.rng.dirichlet((alpha,) * n_comps)

        # -- 2.b sample normal for each component
        n_signal = int(np.ceil((1.0 - noise_ratio) * n_obs))
        comp_obs = [int(np.ceil(w * n_signal)) for w in wts]
        means = self.rng.uniform(0, n_bins, size=n_comps)
        sigmas = self.rng.uniform(*self.sigmas, size=n_comps)
        samples = [self.rng.normal(mi, si, size=ni)
                   for mi, si, ni in zip(means, sigmas, comp_obs)]
        # apply rounding
        rounding = self.rng.choice(self.rounding)
        rounding = min(rounding, n_bins // 5)
        if rounding > 1:
            r = rounding
            samples = [m + np.round((sample - m) / r) * r
                       for sample, m in zip(samples, means)]

        # -- 2.c create counts from samples and add noise
        signal = np.zeros(n_bins, dtype=int)
        for sample in samples:
            for j in np.round(sample).astype(int):
                if 0 <= j < n_bins:
                    signal[j] += 1
        disp_coef = self.rng.uniform(*self.signal_dispersion)
        disp = np.exp(self.rng.normal(0.0, disp_coef, n_bins))
        signal_noisy = np.round(disp * signal)


        # -- 2.d evaluate densities on test points
        xtest = np.arange(n_bins, dtype=float) + 0.5
        signal_dens = sum(w * norm.pdf(xtest, loc=mi, scale=si)
                          for mi, si, w in zip(means, sigmas, wts)) + 1e-10
        signal_normalized = signal_dens / signal_dens.sum()
        total_noise = noise.sum()
        total_obs = signal.sum()
        expected = signal_normalized * sum(signal)

        # 3. return output
        output = {
            'n_bins': n_bins,
            'noise': noise,
            'signal': signal,
            'counts': noise + signal_noisy,
            'means': means,
            'sigmas': sigmas,
            'n_obs': sum(signal),
            'n_bins': n_bins,
            'signal_dens': signal_dens,
            'signal_normalized': signal_normalized,
            'weights': wts,
            'n_comps': n_comps,
            'noise_dispersion': noise_dispersion,
            'signal_overdispersion': disp_coef,
            'noise_ratio': noise_ratio,
            'total_noise': total_noise,
            'total_obs': total_obs,
            'expected': expected,
            'rounding': rounding
        }
        return output

# # %%


# simulator = RecordGenerator()


# # %%

# data = simulator.generate()

# x = np.arange(data['n_bins'])
# plt.bar(x, data['counts'], width=1.0)
# plt.plot(x, data['expected'], c="red")


# # %%
# def normalize(x):
#     x = x / sum(x)
#     return x


# resids = normalize(data['signal_normalized']) - normalize(data['counts'])
# plt.scatter(x, resids)
# plt.axhline(0, c="red")
# plt.title("residuals")

# # %%
