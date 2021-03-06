import numpy as np
import matplotlib.pyplot as plt
from lib_neurips_perso import em_algorithm_single, sinkhorn_em_algorithm_single, error
from gaussian_mixture import sample



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
color = ["blue", "red", "green", "yellow", "purple", "black", "grey", "orange", "pink"]

fig, ax = plt.subplots(4, 4, figsize=(10, 10))
#for k_star, mu_star in enumerate(mu_star_list):
mu_star = 1

for k_0, mu0 in enumerate(np.linspace(0, 2 * mu_star, 16)):
    mu_star_reshaped = np.array([mu_star, 0])
    means = np.array([mu_star_reshaped, -mu_star_reshaped])
    #for i, alpha in tqdm(enumerate(alpha_array[3])):
    alpha = 0.7
    weights = [alpha, 1 - alpha]
    samples, modes = sample(weights, means, covs, N_SAMPLES, 2)
    means_em, weights_sinkhorn, seq_sinkhorn, weights_list = sinkhorn_em_algorithm_single(
        samples = samples,
        mu0 = np.array([np.array([mu0, 0]), np.array([-mu0, 0])]),
        sigma = 1,
        log_theta0= np.array([np.log(weights[0]), np.log(weights[1])]).reshape(2, 1),
        n_iter = 10,
        n_iter_sinkhorn = 50)
    means_em, thetaseq, seq_vanilla, weights_list = em_algorithm_single(
        samples = samples,
        mu0 = np.array([np.array([mu0, 0]), np.array([-mu0, 0])]),
        sigma = 1,
        log_theta0 = np.log(weights).reshape(2, 1),
        n_iter = 10,
        update_theta=False)
    mu_seq_sinkhorn = np.array([seq_sinkhorn[i][0][0] for i in range(len(seq_sinkhorn))])
    mu_seq_vanilla = np.array([seq_vanilla[i][0][0] for i in range(len(seq_vanilla))])

    ax[k_0 // 4][k_0 % 4].plot(range(len(mu_seq_sinkhorn)), np.log(np.abs(mu_star - mu_seq_sinkhorn)), c = "blue", label = "sEM")
    ax[k_0 // 4][k_0 % 4].plot(range(len(mu_seq_vanilla)), np.log(np.abs(mu_star - mu_seq_vanilla)), c = "red", label = "vEM")
    ax[k_0 // 4][k_0 % 4].text(3, -3, r"$\theta_0$ = {}".format(int(mu0 * 100) / 100), fontsize=10)
    #ax[k_star][k_0].plot(range(len(seq)), np.log(np.abs(mu_star - mu_seq)), label = "alpha = {}".format(alpha))
    #ax[k_star][k_0].set_title("theta_star = {}, theta_0 = {}".format(mu_star, mu0))
    #ax[k_star][k_0].set_xlabel("iteration")
    #ax[k_star][k_0].set_ylabel("log error")
    #rho = np.exp(- min(mu_star, mu0)**2)
    #initial_error = np.abs(mu_star - mu0)

    #ax[k_star][k_0].plot(range(len(seq)), [np.log(rho) * i + initial_error for i in range(len(seq))], c="black", label = "upper bound")
plt.legend()
plt.show()



