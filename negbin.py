import numpy as np
from scipy.stats import norm
# import matplotlib.pyplot as plt


class RecordGenerator():
    def __init__(self,
                 seed=None, n_bins=(10, 500), n_obs=(1000, 100000),
                 noise_ratio=(0.001, 0.75), noise_dispersion=(0.05, 2.0),
                 n_comps=(1, 5), alpha_comps=(0.25, 4.0),
                 rounding=[1, 2, 5, 10], max_sigma_to_bins_ratio=1.0,
                 sigmas=(0.001, 25.0), signal_dispersion=(0.00001, 0.4),
                 trim_corners=False):
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
        self.max_sigma_to_bins_ratio = max_sigma_to_bins_ratio
        self.trim_corners = trim_corners

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
        sigmas = np.minimum(sigmas, self.max_sigma_to_bins_ratio * n_bins)
        samples = [self.rng.normal(mi, si, size=ni)
                   for mi, si, ni in zip(means, sigmas, comp_obs)]
        # apply rounding
        rounding = self.rng.choice(self.rounding)
        rounding = min(rounding, max(n_bins // 5, 1))
        if rounding > 1:
            r = rounding
            samples = [m + np.round((sample - m) / r) * r
                       for sample, m in zip(samples, means)]

        # -- 2.c create counts from samples and add noise
        signals = [np.zeros(n_bins, dtype=int) for _ in range(n_comps)]
        for sample, signal in zip(samples, signals):
            for j in np.round(sample).astype(int):
                if 0 <= j < n_bins:
                    signal[j] += 1

        # total signal
        signal = sum(s for s in signals)
        disp_coef = self.rng.uniform(*self.signal_dispersion)
        disp = np.exp(self.rng.normal(0.0, disp_coef, n_bins))
        signal_noisy = np.round(disp * signal)

        # -- 2.d evaluate densities on test points
        xtest = np.arange(n_bins, dtype=float) + 0.5
        signal_denss = [norm.pdf(xtest, loc=mi, scale=si)
                        for mi, si in zip(means, sigmas)]
        # trim the denisty if no observation in corner
        if self.trim_corners:
            for signal_dens, signal in zip(signal_denss, signals):
                j = 0
                while j < len(signal) and signal[j] == 0:
                    signal_dens[j] = 0.0
                    j += 1
                j = n_bins - 1
                while j >= 0 and signal[j] == 0:
                    signal_dens[j] = 0.0
                    j -= 1

        # now must renormalize the signals
        to_remove = []
        comp_counts = []
        for i, dens, cnts in zip(range(n_bins), signal_denss, signals):
            M = cnts.sum()
            if M == 0:
                to_remove.append(i)
            else:
                dens += 1e-10
                dens /= dens.sum()
            comp_counts.append(M)

        if len(to_remove) > 0:
            signal_denss = [s for i, s in enumerate(signal_denss)
                            if i not in to_remove]
            means = [s for i, s in enumerate(means) if i not in to_remove]
            sigmas = [s for i, s in enumerate(sigmas) if i not in to_remove]
            comp_counts = [s for i, s in enumerate(comp_counts)
                           if i not in to_remove]

        modes_onehot = np.zeros(n_bins, dtype=int)
        wts_final = np.array(comp_counts, dtype=float) / sum(comp_counts)
        wts_final_onehot = np.zeros(n_bins)
        for m, w in zip(means, wts_final):
            modes_onehot[int(m)] = 1
            wts_final_onehot[int(m)] = w

        # total signal density
        wts = wts_final
        signal_dens = sum(w * s for w, s in zip(signal_denss, wts))
        try:
            signal_normalized = signal_dens / signal_dens.sum()
        except:
            print('wts: ', wts_final)
            print('signal_denss: ', signal_denss)
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
            'comp_counts': comp_counts,
            'modes_onehot': modes_onehot,
            'wts_final': wts_final,
            'modes_wts': wts_final_onehot,
            'sigmas': sigmas,
            'n_obs': signal.sum(),
            'n_bins': n_bins,
            'signal_dens': signal_dens,
            'signal_comps': signal_denss,
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
