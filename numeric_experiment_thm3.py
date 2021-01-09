import numpy as np
import matplotlib.pyplot as plt
from lib_neurips_perso import em_algorithm_single, sinkhorn_em_algorithm_single, error
from gaussian_mixture import sample
from tqdm import tqdm


dim = 2
mu_star = np.array([1, 0])
means = np.array([mu_star, -mu_star])
covs = [np.eye(2) for i in range(2)]
N_SAMPLES = 3000

step_alpha = 0.01
convergence = {}
algorithms = ["vEM", "sEM", "oEM"]
mu_star_list = [1, 2]#[0.5, 1, 2, 5]
alpha_array = np.linspace(0.5, 0.99, 5)

fig, ax = plt.subplots(2, 2, figsize=(10, 15))
for k_star, mu_star in enumerate(mu_star_list):
    for k_0, mu0 in enumerate([mu_star / 2, mu_star * 2]):
        mu_star_reshaped = np.array([mu_star, 0])
        means = np.array([mu_star_reshaped, -mu_star_reshaped])
        for i, alpha in tqdm(enumerate(alpha_array)):
            weights = [alpha, 1 - alpha]
            samples, modes = sample(weights, means, covs, N_SAMPLES, 2)
            means_em, weights_sinkhorn, seq, weights_list = sinkhorn_em_algorithm_single(
                samples = samples,
                mu0 = np.array([np.array([mu0, 0]), np.array([-mu0, 0])]),
                sigma = 1,
                log_theta0= np.array([np.log(weights[0]), np.log(weights[1])]).reshape(2, 1),
                n_iter = 12,
                n_iter_sinkhorn = 50)
            mu_seq = np.array([seq[i][0][0] for i in range(len(seq))])
            ax[k_star][k_0].plot(range(len(seq)), np.log(np.abs(mu_star - mu_seq)), label = "alpha = {}".format(int(100 * alpha) / 100))
            ax[k_star][k_0].set_title(r"$\theta^* = {}, \theta_0 = {}$".format(mu_star, mu0), fontsize = 20)
            ax[k_star][k_0].set_xlabel("iteration", fontsize = 15)
            ax[k_star][k_0].set_ylabel("log error", fontsize = 15)
            rho = np.exp(- min(mu_star, mu0)**2)
            initial_error = np.abs(mu_star - mu0)

        ax[k_star][k_0].plot(range(len(seq)), [np.log(rho) * i + initial_error for i in range(len(seq))], c="black", label = "upper bound")

plt.legend()
plt.show()



