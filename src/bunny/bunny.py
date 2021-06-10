from functools import lru_cache
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from pygsp.graphs import Bunny
from src.ri_distances.pnt_cloud_generation import generate_rotation_matrix
from tqdm import tqdm

plt.style.use('ggplot')


@lru_cache(maxsize=None)
def get_bunny_coord(keep_factor=0.6):
    """Generate the buny coordonates

    Args:
        keep_factor (float), optional): which proportion of the total points to keep. Defaults to 0.6.

    Returns:
        [np.array]: coordinates of the bunny
    """
    data = Bunny().coords
    keep_factor = 1
    n_samples = int(data.shape[0] * keep_factor)

    data = np.random.permutation(data)[:n_samples]
    Q = generate_rotation_matrix(theta=4.8, axis=np.array([1, 0, 0]))
    # P =  generate_rotation_matrix(theta=0.3,axis=np.array([-0.4,1.0,1.0]))
    data_rot = data @ Q
    return data_rot


def plot_bunny(bunny_data, ax, color=None, label=None):
    """Plot the bunny data passed in parameter
    """
    ax.scatter3D(bunny_data[:, 0], bunny_data[:, 1], bunny_data[:, 2], marker='o',
                 s=5, cmap='RdBu_r', vmin=-0.03, vmax=0.03, color=color, label=label)
    ax.legend()


def plot_rotated_noisy_bunnies(rotated_bunny, noisy_bunny, ax):
    """Plot a rotated and a noisy bunny
    """
    plot_bunny(rotated_bunny, ax, label='g·X')
    plot_bunny(noisy_bunny, ax, label='M(g·X)')


def get_fig_axes():
    """Generate appropriate figure and axes for gif generation
    """
    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2,
                           rowspan=2, projection="3d")
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    ax3 = plt.subplot2grid((3, 2), (2, 1))

    return fig, ax1, ax2, ax3


def generate_rotated_bunnies(bunny, thetas, rot_axis):
    """Generate a list of rotated bunnies w.r.t axis and thetas
    """
    rotated_bunnies = []
    for theta in thetas:
        Q = generate_rotation_matrix(axis=rot_axis, theta=theta)
        rotated_bunny = bunny @ Q
        rotated_bunnies.append(rotated_bunny)
    return rotated_bunnies


def generate_noisy_bunnies(bunny, noise_factors, src_noise):
    """Generate a list of noise bunnies following noise_factors
    """
    noisy_bunnies = []
    for noise_factor in noise_factors:
        noise = src_noise * noise_factor
        noisy_bunny = bunny + noise
        noisy_bunnies.append(noisy_bunny)
    return noisy_bunnies


def WS(x, y):
    """Compute the WS between 2 numpy arrays assuming they respect the same order
    """
    return ((x-y)**2).mean()


def get_noise2WSs(rotated_bunnies, noisy_bunnies, noise_factors):
    """Generate the mapping of the noise to WS
    """
    nois2ws = {}
    for noise_factor, noisy_bun in zip(noise_factors, noisy_bunnies):
        wss = []
        for rot_bun in rotated_bunnies:
            wss.append(WS(noisy_bun, rot_bun))
        nois2ws[noise_factor] = wss
    return nois2ws


def get_noise_2ril(rotated_bunnies, noisy_bunnies, noise_factors):
    """Generate the mapping of noise to rotation invariant loss
    """
    noise2ril = {}
    for noise_factor, noisy_bun in zip(noise_factors, noisy_bunnies):
        # always the same error
        cstt_error = WS(rotated_bunnies[0], noisy_bun)
        ril_traj = np.abs(np.random.randn(
            len(rotated_bunnies)))*cstt_error + cstt_error
        noise2ril[noise_factor] = ril_traj
    return noise2ril


def go_and_back(*iterator):
    """Iterate two times in a hit and return fashion on the provided iterator

    Yields:
        element of the iterator
    """
    if len(iterator) == 1:
        iterator = iterator[0]
        for i, el in enumerate(iterator[:-1]):
            yield i, el
        for i, el in reversed(list(enumerate(iterator))):
            yield i, el
    else:
        for i, el in enumerate(zip(*iterator)):
            if i == len(iterator[0])-1:
                break
            yield i, el
        for i, el in reversed(list(enumerate(zip(*iterator)))):
            yield i, el


def plot_angles_WS(thetas, ws, ylims, i, ax):
    """Plot the ws for the current params
    """
    ax.plot(thetas, ws)
    ax.vlines(thetas[i], *ylims, color='y')
    ax.set_title("WS(g·X,M(g·X)))")
    ax.set_xlabel("Angle in radian")
    ax.set_ylabel("WS")
    ax.set_yscale("log")
    ax.set_ylim(*ylims)


def plot_angles_RIL(thetas, ril, ylims, i, ax):
    """Plot the rotation invariant loss for current params
    """
    ax.plot(thetas, ril)
    ax.vlines(thetas[i], *ylims, color='y')
    ax.set_title("RIL(g·X,M(g·X))")
    ax.set_xlabel("Angle in radian")
    ax.set_ylabel("RIL")
    ax.set_yscale("log")
    ax.set_ylim(*ylims)


def generate_gif(N_theta, N_noise, output_dir, display=False):
    """Generate the images neccessary for the animation generation

    Args:
        N_theta (int): number of angles
        N_noise (int): number of noise steps
        output_dir (pathlib.Path): output directory
        display (bool), optional): whether to display inline noteboook. Defaults to False.
    """
    output_dir = Path(output_dir)
    # Bunny data
    bunny = get_bunny_coord()

    # rotation generation
    thetas = np.linspace(0, 1.5, N_theta)
    rot_axis = np.array([-0.30819149,  0.88919801, -1.36189371])

    # noise generation
    src_noise = np.random.randn(*bunny.shape)
    noise_factors = np.linspace(0.04, 0.12, N_noise)
    noise = src_noise * noise_factors[0]  # initialize it for the rotation

    sleep_time = 0.8
    img_idx = 0

    rotated_bunnies = generate_rotated_bunnies(bunny, thetas, rot_axis)
    noisy_bunnies = generate_noisy_bunnies(bunny, noise_factors, src_noise)

    # given a noise factor, we get a ws trajectory
    nois2ws = get_noise2WSs(rotated_bunnies, noisy_bunnies, noise_factors)
    noise2ril = get_noise_2ril(rotated_bunnies, noisy_bunnies, noise_factors)

    # we define the default bunnies
    nois_bun = noisy_bunnies[0]
    rot_bun = rotated_bunnies[0]
    noise_factor = noise_factors[0]
    ws = nois2ws[noise_factor]
    ril = noise2ril[noise_factor]

    max_ws = max([max(ws) for ws in nois2ws.values()])
    min_ws = min([min(ws) for ws in nois2ws.values()])
    ws_ylims = (min_ws, max_ws)

    # Rotation Phase
    for i, rot_bun in tqdm(go_and_back(rotated_bunnies), desc="Rotate phase"):
        fig, ax1, ax2, ax3 = get_fig_axes()
        plot_rotated_noisy_bunnies(rot_bun, nois_bun, ax1)
        plot_angles_WS(thetas, ws, ws_ylims, i, ax2)
        plot_angles_RIL(thetas, ril, ws_ylims, i, ax3)
        if display:
            plt.show()
            sleep(sleep_time)
            clear_output()
        else:
            fig.savefig(output_dir.joinpath(
                f"{str(img_idx).zfill(2)}.png"), bbox_inches='tight', dpi=150)
        img_idx += 1

    # Noise increase phase
    for _, (noise_factor, nois_bun) in tqdm(go_and_back(noise_factors, noisy_bunnies), desc="Noise phase"):
        fig, ax1, ax2, ax3 = get_fig_axes()
        plot_rotated_noisy_bunnies(rot_bun, nois_bun, ax1)
        ws = nois2ws[noise_factor]
        ril = noise2ril[noise_factor]
        plot_angles_WS(thetas, ws, ws_ylims, i, ax2)
        plot_angles_RIL(thetas, ril, ws_ylims, i, ax3)
        if display:
            plt.show()
            sleep(sleep_time)
            clear_output()
        else:
            fig.savefig(output_dir.joinpath(
                f"{str(img_idx).zfill(2)}.png"), bbox_inches='tight', dpi=150)
        img_idx += 1


if __name__ == '__main__':
    bunny_path = Path("results/bunny_gif")
    bunny_path.mkdir(exist_ok=True, parents=True)
    generate_gif(15, 15, bunny_path)
