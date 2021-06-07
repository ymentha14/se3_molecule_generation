import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def get_P(d, L):
    """
    Generate the matrix of projections for sliced gromov wasserstein
    """
    res = np.random.randn(d, L)
    res /= np.sqrt(np.sum(res**2, 0, keepdims=True))
    return res


def get_rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def make_2d_spiral(n_samples, noise=.5):
    """
    Generate a 2d spiral with n_samples and noise amplitude
    """
    n = np.sqrt(np.random.rand(n_samples, 1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples, 1) * noise
    return np.array(np.hstack((d1x, d1y)))


def get_data(n_samples, theta, scale=1, transla=0):
    Xs = make_2d_spiral(n_samples=n_samples, noise=1)-transla
    Xt = make_2d_spiral(n_samples=n_samples, noise=1)
    A = get_rot(theta)

    Xt = (np.dot(Xt, A))*scale+transla

    return Xs, Xt


def plot_perf(nlist, err, color, label, errbar=False, perc=20, ax=None):
    ax.plot(nlist, err.mean(0), label=label, color=color)
    if errbar:
        plt.fill_between(nlist, np.percentile(err, perc, axis=0), np.percentile(err, 100-perc, axis=0),
                         alpha=0.2, facecolor=color)


# def get_RISGW_3d_err_diff(angles, n_proj, nbloop, noise_factor=0.15, max_iter=100):
#     """
#     Compute the difference of metric between RISGW of a spiral point cloud and its
#     noisy version and RISGW of the same point cloud and its noisy+rotated version.
#     Ideally, these metrics should be constant at 0, however if the algorithm turns out
#     to be suboptimal, the difference will be greater than 0

#     Args:
#         angles (np.array): list of angles to induce rotations for (in radian)
#         L (int): number of projections to use for sgw
#         nbloop (int): number of times to run the experiment for
#         noise_factor (float, optional): noise factor for the spiral target

#     Returns:
#         [np.array]: 2d array (nbloop x len(angles)) computed risgw distances
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     risgw_results = []
#     for i in range(nbloop):
#         results = []
#         P3 = torch.from_numpy(get_P(3, n_proj)).to(torch.float32)
#         spiral, _ = get_src_scaled_spirals()

#         for j, theta in enumerate(tqdm(angles)):
#             # add the noise to the spiral
#             noise = torch.randn(spiral.shape) * noise_factor
#             noisy_spiral = spiral + noise

#             # we compute the RISGW on the noisy spiral
#             ref_metric = risgw_gpu(
#                 spiral.squeeze(0),
#                 noisy_spiral.squeeze(0),
#                 device=device,
#                 P=P3,
#                 max_iter=max_iter,
#                 verbose=False)
#             Q = get_rotation_matrix(theta=theta)
#             rot_noisy_spiral = noisy_spiral @ Q

#             rot_metric = risgw_gpu(
#                 spiral.squeeze(0),
#                 rot_noisy_spiral.squeeze(0),
#                 device=device,
#                 P=P3,
#                 max_iter=max_iter,
#                 verbose=False)

#             results.append((rot_metric, ref_metric))
#         risgw_results.append(results)
#     return np.array(risgw_results)


def plot_err_diff(angles, risgw_results, ax):
    """
    Plot the error difference as computed in src.se3.sgw.get_RISGW_3d_err_diff
    """

    plot_perf(angles, risgw_results, 'k', 'RISGW', True, ax=ax)

    ax.set_title("Difference of RISGW btwn input/rot+nois and input/nois (3D)")
    ax.set_xlabel('Rotation angle (radian)')
    ax.set_ylabel('Difference')
    ax.legend()
