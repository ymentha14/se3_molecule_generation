"""Point cloud generation functions to test benchmarks of rotation invariant functions
"""

import numpy as np
import torch
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R


def to_torch_tensor(np_array):
    """
    convert a numpy array to a torch tensor by putting it to the appropriate type
    and dimension
    """
    return torch.tensor(np_array).unsqueeze(0).to(torch.float32)


def to_numpy_array(torch_tensor):
    """
    convert a torch tensor to a numpy array by putting it to the appropriate type
    and dimension
    """
    return torch_tensor.squeeze().detach().numpy()


# Rotation and Permutation matrices
def generate_permutation_matrix(N):
    """
    Generate a random permutation matrix
    """
    return np.random.permutation(np.identity(N))


def generate_rotation_matrix(theta=None, axis=None):
    """
    Return a rotation matrix corresponding to a rotation of amplitude theta
    around the provided axis. If an argument is set to None, a random value is assigned
    """
    if theta is None:
        theta = np.random.rand(1)[0] * 2 * np.pi
    if axis is None:
        axis = np.random.rand(3) - 0.5
        axis /= norm(axis)
    return Rotation.from_rotvec(theta * axis).as_matrix()


def rotate(*args):
    """
    Return the rotated version of each tensor passed in parameter using the same rotation
    matrix

    """
    Q = generate_rotation_matrix()
    return [points_tens @ Q for points_tens in args]


# Centering
def center_batch(batch):
    """center a batch

    Args:
        batch (torch.tensor): [n_batch x N_points x 1 x 3]

    Returns:
        [type]: [description]
    """
    mean_input = batch.mean(axis=1).unsqueeze(1)
    return batch - mean_input


def center(sample):
    """
    Center batch to have a barycenter around 0

    Args:
        sample (np.array): N x 3 sample
    """
    return sample - sample.mean(axis=0)


def get_points_on_sphere(N):
    """
    Generate N evenly distributed points on the unit sphere centered at
    the origin. Uses the 'Golden Spiral'.
    Code by Chris Colbert from the numpy-discussion list.
    """
    phi = (1 + np.sqrt(5)) / 2  # the golden ratio
    long_incr = 2 * np.pi / phi  # how much to increment the longitude

    dz = 2.0 / float(N)  # a unit sphere has diameter 2
    bands = np.arange(N)  # each band will have one point placed on it
    z = bands * dz - 1 + (dz / 2)  # the height z of each band/point
    r = np.sqrt(1 - z * z)  # project onto xy-plane
    az = bands * long_incr  # azimuthal angle of point modulo 2 pi
    x = r * np.cos(az)
    y = r * np.sin(az)
    return np.array((x, y, z)).transpose()


def get_angle(u, v):
    """
    Return the angle between 3d vectors u and v
    """
    cos_angle = np.dot(u, v) / norm(u) / norm(v)
    return np.arccos(np.clip(cos_angle, -1, 1))


def get_n_regular_rotations(N):
    """
    Return N regular rotations matrices

    Args:
        N (int): number of rotation matrixes to return
    """
    # generate random points on the unit sphere
    reg_points = get_points_on_sphere(N)

    # take the first vector as the reference one
    ref_vec = reg_points[0]

    # computation of angle and cross product as described here
    # https://math.stackexchange.com/questions/2754627/rotation-matrices-between-two-points-on-a-sphere
    thetas = [get_angle(ref_vec, v) for v in reg_points]
    cross_vecs = np.cross(ref_vec, reg_points)
    rotations = [
        (R.from_rotvec(theta * cross_vec).as_matrix())
        for theta, cross_vec in zip(thetas, cross_vecs)
    ]
    return rotations


# Point Cloud Generation


def generate_target(src, theta=None, noise_factor=0.0, permute=True):
    N = src.shape[0]
    Q = generate_rotation_matrix(theta=theta)
    if permute:
        P = generate_permutation_matrix(N)
    else:
        P = np.eye(N)
    target = src @ Q
    target = P @ target
    target += np.random.randn(*target.shape) * noise_factor
    return Q, P, target


# Gaussian Point cloud


def get_gaussian_point_cloud(N_pts):
    return np.random.rand(N_pts, 3)


# Spiral Point cloud


def get_asym_spiral(spiral_amp=1.0, N_pts=40):
    """
    Generate a spiral with the given amplitude

    Args:
        spiral_amp (float, optional): spiral amplitude

    Returns:
        np.array: spiral point cloud
    """
    zline = np.linspace(0, spiral_amp, N_pts)
    xline = [(i + 4) * np.sin(i) / 10 for i in zline * 4 * np.pi]
    yline = [(i + 4) * np.cos(i) / 10 for i in zline * 4 * np.pi]
    return np.array([xline, yline, zline]).transpose()


def get_spiral(spiral_amp=1.0, N_pts=40):
    """
    Generate a spiral with the given amplitude

    Args:
        spiral_amp (float, optional): spiral amplitude

    Returns:
        np.array: spiral point cloud
    """
    zline = np.linspace(0, spiral_amp, 40)
    xline = np.sin(zline * 4 * np.pi)
    yline = np.cos(zline * 4 * np.pi)
    return np.array([xline, yline, zline]).transpose()

def get_custom_spiral(spiral_amp=1.0,scaling=1.0,shift=0.0,asym=False,centering=False):
    assert(not (centering and shift!=0))

    if asym:
        points = get_asym_spiral(spiral_amp=spiral_amp)
    else:
        points = get_spiral(spiral_amp=spiral_amp)

    [xline, yline, zline] = points.transpose()
    zline *= scaling
    zline += shift
    points = np.array([xline, yline, zline ]).transpose()
    if centering:
        points = center(points)
    return points

def get_src_shifted_spirals(
    spiral_amp=1.0, shift=0.5, asym=False, center_input=False, center_target=False
):
    """
    Return vertical src spiral cloud point and its vertically shifted version.
    """
    # torch.set_default_dtype(torch.float64) # works best in float64
    if asym:
        points = get_asym_spiral(spiral_amp=spiral_amp)
    else:
        points = get_spiral(spiral_amp=spiral_amp)
    target_points = np.array([xline, yline, (zline + shift)]).transpose()
    if center_input:
        points = center(points)
    if center_target:
        target_points = center(target_points)
    return points, target_points


def get_src_scaled_spirals(
    spiral_amp=1.0, z_scale=3, asym=False, center_input=False, center_target=False
):
    if asym:
        points = get_asym_spiral(spiral_amp=spiral_amp)
    else:
        points = get_spiral(spiral_amp=spiral_amp)
    [xline, yline, zline] = points.transpose()
    target_points = np.array([xline, yline, zline * z_scale]).transpose()
    if center_input:
        points = center(points)
    if center_target:
        target_points = center(target_points)
    return points, target_points


def get_src_inverted_spirals(spiral_amp=1.0):
    """
    Return inverted spiral with same positions
    """
    points = get_spiral(spiral_amp=spiral_amp)
    [xline, yline, zline] = points
    target_points = np.array([-xline, yline, zline]).transpose()

    return points, target_points
