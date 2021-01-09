import numpy as np
import sklearn
import matplotlib.pyplot as plt
from lib_neurips_perso import em_algorithm_single, sinkhorn_em_algorithm_single
import celluloid


def sample(weights, means, covs, n_samples, dim):
    samples = np.zeros((n_samples, dim))
    modes = np.zeros(n_samples, int)
    for i in range(n_samples):
        mode = np.random.choice(list(range(len(weights))), 1, p = weights)[0] # select the mode generating the sample
        if dim > 1:
            samples[i] = np.random.multivariate_normal(means[mode], covs[mode])
        else:
            samples[i] = np.random.normal(means[mode], covs[mode])
        modes[i] = mode
    return samples, modes

#def sample1d(weights, means, vars, n_samples):
 #   samples = np.zeros(n_samples)
 #   modes

if __name__ == '__main__':
    weights = [0.2, 0.8]

    dim = len(weights)

    mu_star = np.array([2, 0])

    means = np.array([-mu_star, mu_star])


    covs = [np.eye(2) for i in range(2)]
    N_SAMPLES = 1000
    samples, modes = sample(weights, means, covs, N_SAMPLES, 2)
    colors = np.array(["blue", "red"])
    plt.scatter(x = samples[:, 0], y = samples[:, 1], c = colors[modes], alpha = 0.2)

    #plt.scatter(x = samples[:, 0], y = [0.1 * np.random.random() for _ in range(N_SAMPLES)], c = colors[modes], alpha = 0.2)

    plt.scatter(x = means[:, 0], y = means[:, 1], c = "black")

    #plt.scatter(x = means[:, 0], y = [1 for _ in len(weights)], c = "black")

    #plt.scatter(x = means, y = [0.05 for _ in range(len(weights))], c = "black")


    plt.show()

