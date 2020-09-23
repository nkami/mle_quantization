import numpy as np
from numpy.random import default_rng
import scipy
import scipy.stats as stats
from scipy import integrate
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_pdf(sample_size: int, distribution: str):
    if distribution is 'normal':
        def pdf(x):
            return x * sample_size * (stats.norm.cdf(x) ** (sample_size - 1)) * stats.norm.pdf(x)
    elif distribution is 'laplace':
        def pdf(x):
            return x * sample_size * (stats.laplace.cdf(x) ** (sample_size - 1)) * stats.laplace.pdf(x)
    else:
        pdf = None
    return pdf


def calc_optimal_beta(alpha: float, distribution: str, s: float, n_bits: int):
    betas = np.linspace(1., 16., num=80)
    betas_probs = []
    split_val = 2 ** (n_bits - 1)
    for beta in betas:
        dense_locs = np.linspace(start=0., stop=split_val - 1, num=split_val)
        sparse_locs = np.linspace(start=split_val, stop=(split_val + (split_val - 1) * beta), num=split_val)
        locs = np.concatenate([dense_locs, sparse_locs])
        normalize_const = (s / (split_val + (split_val - 1) * beta))
        xqs = normalize_const * locs
        start_bins = (xqs / (1 + alpha))
        end_bins = (xqs / (1 - alpha))
        sorted_indx = np.argsort(start_bins)
        start_bins = np.take(start_bins, sorted_indx)
        end_bins = np.take(end_bins, sorted_indx)
        merged = [[start_bins[0], end_bins[0]]]
        for current in range(start_bins.shape[0]):
            previous = merged[-1]
            if start_bins[current] <= previous[1]:
                previous[1] = max(previous[1], end_bins[current])
            else:
                merged.append([start_bins[current], end_bins[current]])
        del merged[0]
        merged.insert(0, [0, normalize_const * 0.5])
        merged = np.asarray(merged)
        if distribution is 'normal':
            probs = stats.norm.cdf(merged[:, 1]) - stats.norm.cdf(merged[:, 0])
            probs[0] *= 0.1
            betas_probs.append(np.sum(probs))
        elif distribution is 'laplace':
            probs = stats.laplace.cdf(merged[:, 1]) - stats.laplace.cdf(merged[:, 0])
            probs[0] *= 0.1
            betas_probs.append(np.sum(probs))
        else:
            return None
    return betas, betas_probs, betas[int(np.argmax(betas_probs))]


def plot_non_uniform_graph():
    points = np.linspace(0., 13, num=100)
    laplace_vals = stats.laplace.pdf(points)
    plt.plot(points, laplace_vals)
    plt.ylabel('Probability')
    beta = 2.5
    dense_locs = [i for i in range(4)]
    sparse_locs = [4 + i * beta for i in range(5)]
    bins_locs = dense_locs + sparse_locs
    dense_names = [str(loc) for loc in dense_locs]
    sparse_names = ['4', '4 + ' + r'$ \beta $'] + ['4 + ' + str(i) + r'$ \beta $' for i in range(2, 4)]
    bins_names = dense_names + sparse_names
    plt.xticks(bins_locs, bins_names)
    for prev, cur in zip(bins_locs[:-1], bins_locs[1:]):
        plt.axvline(x=prev + ((cur - prev) / 2), ymin=0, ymax=0.2, linestyle='--', color='orange')
    plt.show()


if __name__ == '__main__':
    alpha = 0.01
    n_bits = 4
    distribution = 'laplace'
    avg_sample_size = 25 * 1000
    num_experiments = 10000
    optimal_betas = []
    rng = default_rng()
    pdf = get_pdf(avg_sample_size, distribution)
    avg_s, err = integrate.quad(pdf, 0., float('inf'))
    betas, probs, avg_opt = calc_optimal_beta(alpha, distribution, avg_s, n_bits)
    for _ in tqdm(range(num_experiments)):
        s = np.amax(rng.laplace(0., 1., avg_sample_size))
        optimal_betas.append(calc_optimal_beta(alpha, distribution, s, n_bits)[2])
    optimal_betas = np.asarray(optimal_betas)
    mle_beta = optimal_betas[np.argmax(np.unique(optimal_betas, return_counts=True)[1])]
    plt.plot(betas, probs, 'b')
    plt.plot(mle_beta, probs[np.argmax(betas == mle_beta)], 'or')
    plt.xlabel(r'$ \beta $')
    plt.ylabel('Probability')
    plt.legend(['For Average S = {0:.2f}'.format(avg_s), 'Optimal Simulated Value = {0:.2f}'.format(mle_beta)])
    plt.show()

