import numpy as np
import sklearn
import matplotlib.pyplot as plt
from lib_neurips_perso import em_algorithm_single, sinkhorn_em_algorithm_single, error
import celluloid
from gaussian_mixture import sample
import ot
#print(weights)
#print(weights_sinkhorn)
alpha = 0.2
weights = np.array([alpha, 1 - alpha])

dim = len(weights)

mu_star = np.array([2, 0])

means = np.array([mu_star, -mu_star])


covs = [np.eye(2) for i in range(2)]
N_SAMPLES = 1000
samples, modes = sample(weights, means, covs, N_SAMPLES, 2)



# means_em, thetaseq, seq, weights_list = em_algorithm_single(
#     samples = samples,
#     mu0 = np.array([np.array([-4, 0]), np.array([4, 0])]),
#     sigma = 1,
#     log_theta0= np.array([np.log(weights[0]), np.log(weights[1])]).reshape(2, 1),
#     n_iter = 4,
#     update_theta=False)
#
means_em, weights_sinkhorn, seq, weights_list = sinkhorn_em_algorithm_single(
    samples = samples,
    mu0 = np.array([np.array([-2, 0]), np.array([2, 0])]),
    sigma = 1,
    log_theta0= np.array([np.log(weights[0]), np.log(weights[1])]).reshape(2, 1),
    n_iter = 1,
    n_iter_sinkhorn = 50)
thetaseq = [np.array(weights).reshape(2, 1) for i in range(len(seq))]

print(means_em)
print(means)
print(error(means_em, means, weights))
print(weights)
print(ot.emd2_1d(means_em[:, 0], means[:, 0], thetaseq[-1].reshape(-1), weights.reshape(-1)))

animate = False

if animate:

    fig = plt.figure()
    camera = celluloid.Camera(fig)
    cm = "viridis"

    #intital setup
    proba_mode = (weights_list[0][0] / (weights_list[0][0] + weights_list[0][1]))
    plt.scatter(x = samples[:, 0], y = samples[:, 1], c = proba_mode, cmap = cm, alpha = 0.2)
    plt.scatter(x = means[0][0], y = means[0][1], c = "black", s = 200 * weights[0])
    plt.scatter(x = means[1][0], y = means[1][1], c = "black", s = 200 * weights[1])
    plt .scatter(seq[0][0][0], seq[0][0][1], c = "red", s = 200 * thetaseq[0][0], alpha = 0.5)
    plt.scatter(seq[0][1][0], seq[0][1][1], c = "red", s = 200 * thetaseq[0][1], alpha = 0.5)
    camera.snap()


    for i in range(1, len(weights_list)):
        #print(weights_list[i][:,3][0] / (weights_list[i][:,3][0] + weights_list[i][:,3][1]))
        #E step
        proba_mode = (weights_list[i][0] / (weights_list[i][0] + weights_list[i][1]))
        #plt.scatter(x = samples[:, 0], y = samples[:, 1], c = colors[modes], alpha = 0.2)
        plt.scatter(x = samples[:, 0], y = samples[:, 1], c = proba_mode, alpha = 0.2, cmap = cm)
        plt.text(0, 3, "E", fontsize = 30)
        #previous means
        plt.scatter(x = means[0][0], y = means[0][1], c = "black", s = 200 * weights[0])
        plt.scatter(x = means[1][0], y = means[1][1], c = "black", s = 200 * weights[1])
        plt .scatter(seq[i-1][0][0], seq[i-1][0][1], c = "red", s = 200 * thetaseq[i-1][0], alpha = 0.5)
        plt.scatter(seq[i-1][1][0], seq[i-1][1][1], c = "red", s = 200 * thetaseq[i-1][1], alpha = 0.5)
        camera.snap()
        #plt.scatter(x = means[:, 0], y = means[:, 1], c = "black")
        #M step
        plt.scatter(x = samples[:, 0], y = samples[:, 1], c = proba_mode, alpha = 0.2, cmap = cm)
        plt.text(0, 3, "M", fontsize = 30)
        plt.scatter(x = means[0][0], y = means[0][1], c = "black", s = 200 * weights[0])
        plt.scatter(x = means[1][0], y = means[1][1], c = "black", s = 200 * weights[1])
        plt.scatter(seq[i][0][0], seq[i][0][1], c = "red", s = 200 * thetaseq[i][0], alpha = 0.5, edgecolor = "white")
        plt.scatter(seq[i][1][0], seq[i][1][1], c = "red", s = 200 * thetaseq[i][1], alpha = 0.5, edgecolor = "white")
        camera.snap()

    animation = camera.animate()
    animation.save('animation.gif', fps = 0.5)

else:
    n_cols = 2
    fig, ax = plt.subplots(len(weights_list), n_cols, figsize=(10, 15))
    cm = "viridis"

    #intital setup
    proba_mode = (weights_list[0][0] / (weights_list[0][0] + weights_list[0][1]))
    ax[0][0].scatter(x = samples[:, 0], y = samples[:, 1], c = proba_mode, cmap = cm, alpha = 0.2)
    ax[0][0].scatter(x = means[0][0], y = means[0][1], c = "black", s = 200 * weights[0])
    ax[0][0].scatter(x = means[1][0], y = means[1][1], c = "black", s = 200 * weights[1])
    ax[0][0].scatter(seq[0][0][0], seq[0][0][1], c = "red", s = 200 * thetaseq[0][0], alpha = 0.5)
    ax[0][0].scatter(seq[0][1][0], seq[0][1][1], c = "red", s = 200 * thetaseq[0][1], alpha = 0.5)
    ax[0][0].set_title("E step", fontsize = 20)
    ax[0][1].set_title("M step", fontsize = 20)

    for i in range(1, len(weights_list)):
        #print(weights_list[i][:,3][0] / (weights_list[i][:,3][0] + weights_list[i][:,3][1]))
        #E step
        proba_mode = (weights_list[i][0] / (weights_list[i][0] + weights_list[i][1]))
        #plt.scatter(x = samples[:, 0], y = samples[:, 1], c = colors[modes], alpha = 0.2)
        ax[i][0].scatter(x = samples[:, 0], y = samples[:, 1], c = proba_mode, alpha = 0.2, cmap = cm)
        #ax[i][0].text(0, 3, "E", fontsize = 30)
        #previous means
        ax[i][0].scatter(x = means[0][0], y = means[0][1], c = "black", s = 200 * weights[0])
        ax[i][0].scatter(x = means[1][0], y = means[1][1], c = "black", s = 200 * weights[1])
        ax[i][0] .scatter(seq[i-1][0][0], seq[i-1][0][1], c = "red", s = 200 * thetaseq[i-1][0], alpha = 0.5)
        ax[i][0].scatter(seq[i-1][1][0], seq[i-1][1][1], c = "red", s = 200 * thetaseq[i-1][1], alpha = 0.5)
        #ax[i][0].scatter(x = means[:, 0], y = means[:, 1], c = "black")
        #M step
        ax[i][1].scatter(x = samples[:, 0], y = samples[:, 1], c = proba_mode, alpha = 0.2, cmap = cm)
        #ax[i][1].text(0, 3, "M", fontsize = 30)
        ax[i][1].scatter(x = means[0][0], y = means[0][1], c = "black", s = 200 * weights[0])
        ax[i][1].scatter(x = means[1][0], y = means[1][1], c = "black", s = 200 * weights[1])
        ax[i][1].scatter(seq[i][0][0], seq[i][0][1], c = "red", s = 200 * thetaseq[i][0], alpha = 0.5)
        ax[i][1].scatter(seq[i][1][0], seq[i][1][1], c = "red", s = 200 * thetaseq[i][1], alpha = 0.5)
    plt.show()