import numpy as np
import matplotlib.pyplot as plt
from lib_neurips_perso import em_algorithm_single, sinkhorn_em_algorithm_single, error
from gaussian_mixture import sample
from tqdm import tqdm
import ot


def unique_local_minimum(means_star, alpha, algo):
    weights = np.array([alpha, 1 - alpha])
    for mu_0 in np.linspace(- 10 * means_star, 10 * means_star, 10):
        #mu_0 = np.array([mu_0, -mu_0])
        if algo == "sEM":
            means_em, weights_sinkhorn, seq, weights_list = sinkhorn_em_algorithm_single(
                samples = samples,
                mu0 = mu_0,
                sigma = 1,
                log_theta0 = np.log(weights).reshape(2, 1),
                n_iter = 10,
                n_iter_sinkhorn = 20)
            thetaseq = [np.array(weights).reshape(2, 1) for i in range(len(seq))]

        else:
            means_em, thetaseq, seq, weights_list = em_algorithm_single(
                samples = samples,
                mu0 = mu_0,
                sigma = 1,
                log_theta0 = np.log(weights).reshape(2, 1),
                n_iter = 10,
                update_theta=(algo == "oEM"))

        error = ot.emd2_1d(means_em[:, 0], means_star[:, 0], thetaseq[-1].reshape(-1), weights.reshape(-1))
        if error > 1:
            return False
    return True




dim = 2
mu_star = np.array([1, 0])
means = np.array([mu_star, -mu_star])
covs = [np.eye(2) for i in range(2)]
N_SAMPLES = 1000

step_alpha = 0.01
convergence = {}
algorithms = ["vEM", "sEM", "oEM"]
mu_star = [0.5, 1, 2, 5]
alpha_array = np.linspace(0.5, 0.99, 20)
for algo in algorithms:
    convergence[algo] = np.zeros(len(alpha_array))

fig, ax = plt.subplots(2, 2, figsize=(15, 15))
for k, mu_star in enumerate(mu_star):
    mu_star_reshaped = np.array([mu_star, 0])
    means = np.array([mu_star_reshaped, -mu_star_reshaped])
    for algo in algorithms:
        print(algo)
        for i, alpha in tqdm(enumerate(alpha_array)):
            weights = [alpha, 1 - alpha]
            samples, modes = sample(weights, means, covs, N_SAMPLES, 2)
            weights = [alpha, 1 - alpha]
            unique_min = unique_local_minimum(means, alpha, algo)
            convergence[algo][i] = unique_min
    for i, algo in enumerate(algorithms):
        colors = ["green" if conv == True else "red" for conv in convergence[algo]]
        ax[k // 2][k % 2].scatter(x = alpha_array, y = [i for _ in alpha_array], c=colors)
        ax[k // 2][k % 2].text(0.5, i + 0.1, algo, fontsize = 15)
        ax[k // 2][k % 2].set_xlabel("alpha")
        ax[k // 2][k % 2].set_title(r"$\theta^*$ = {}".format(mu_star), fontsize = 20)

#plt.title("Unique local minimum depending on alpha")
plt.show()

print(convergence)



