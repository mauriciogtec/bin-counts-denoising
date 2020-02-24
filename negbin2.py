import numpy as np
from scipy.stats import norm
# import matplotlib.pyplot as plt
# import pdb


class RecordGenerator():
    def __init__(self,
                 seed=None, n_bins=(10, 500), n_obs=(1000, 100000),
                 noise_ratio=(0.001, 0.75), noise_dispersion=(0.05, 2.0),
                 n_comps=(1, 5), alpha_comps=(0.25, 4.0),
                 rounding=[1, 2, 5, 10], max_sigma_to_bins_ratio=1.0,
                 sigmas=(0.1, 25.0), signal_dispersion=(0.00001, 0.4),
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

        bincounts = []
        pdfs = []
        comp_counts = []
        xbins = np.arange(n_bins, dtype=float) + 0.5
        rounding = min(self.rng.choice(self.rounding), max(n_bins // 5, 1))
        for mi, si, ni in zip(means, sigmas, comp_obs):
            while True:
                sample = self.rng.normal(mi, si, size=ni)
                if rounding > 1:
                    r = rounding
                    sample = mi + np.round((sample - mi) / r) * r
                counts = np.zeros(n_bins, dtype=int)
                for j in np.round(sample).astype(int):
                    if 0 <= j < n_bins:
                        counts[j] += 1
                M = counts.sum()
                if M > 0:
                    comp_counts.append(M)
                    break
            dens = norm.pdf(xbins, loc=mi, scale=si)
            # evaluate and trim density, necessary after rounding
            if self.trim_corners:
                j = 0
                while j < n_bins and counts[j] == 0:
                    dens[j] = 0.0
                    j += 1
                j = n_bins - 1
                while j >= 0 and counts[j] == 0:
                    dens[j] = 0.0
                    j -= 1
                # normalize density as probability
                if dens.sum() == 0:   # std too small! 
                    dens[int(mi)] = 1.0

            # add to data
            bincounts.append(counts)
            pdfs.append(dens)
        comp_counts = np.array(comp_counts, dtype=int)

        # total signal
        signal = sum(s for s in bincounts)
        disp_coef = self.rng.uniform(*self.signal_dispersion)
        disp = np.exp(self.rng.normal(0.0, disp_coef, n_bins))
        signal_noisy = np.round(disp * signal)
        obs = signal_noisy + noise

        # modes and post-trimming weights
        means_onehot = np.zeros(n_bins, dtype=int)
        wts_final = np.array(comp_counts, dtype=float) / sum(comp_counts)
        wts_final_onehot = np.zeros(n_bins)
        for m, w in zip(means, wts_final):
            means_onehot[int(m)] = 1
            wts_final_onehot[int(m)] = w

        # total signal density
        pdf = sum(w * s for w, s in zip(pdfs, wts_final))
        pdf /= pdf.sum()
        total_noise = noise.sum()
        total_obs = comp_counts.sum()
        expected = pdf * total_obs

        # 3. return output
        output = {
            'n_bins': n_bins,
            'noise': noise,
            'signal': signal,
            'signal_noisy': signal_noisy,
            'obs': obs,
            'means': means,
            'comp_counts': comp_counts,
            'means_onehot': means_onehot,
            'wts_final': wts_final,
            'means_wts': wts_final_onehot,
            'sigmas': sigmas,
            'n_obs': total_obs,
            'n_bins': n_bins,
            'pdf': pdf,
            'comp_pdfs': pdfs,
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
