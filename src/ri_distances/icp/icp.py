from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.ri_distances.pnt_cloud_generation import generate_rotation_matrix
from src.ri_distances.rotation_predictor import RotationPredictor


@dataclass
class IcpPredictor(RotationPredictor):
    """
    Predictor object to align point clouds
    """
    max_iter: int = 20
    tolerance: float = 0.001
    N_rots: int = 10

    def best_fit_transform(self, A, B):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
        A: Nxm numpy array of corresponding points
        B: Nxm numpy array of corresponding points
        Returns:
        T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R: mxm rotation matrix
        t: mx1 translation vector
        '''

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R, centroid_A.T)

        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t

    def nearest_neighbor(self, src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''

        assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    def icp(self, A, B, init_pose=None, max_iter=20, tolerance=0.001):
        '''
        The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
        Input:
            A: Nxm numpy array of source mD points
            B: Nxm numpy array of destination mD point
            init_pose: (m+1)x(m+1) homogeneous transformation
            max_iter: exit algorithm after max_iter
            tolerance: convergence criteria
        Output:
            R: mxm rotation matrix
            t: mx1 translation vector
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
        '''
        assert A.shape == B.shape

        # get number of dimensions
        N = A.shape[0]
        m = A.shape[1]

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m+1, N))
        dst = np.ones((m+1, N))
        src[:m, :] = np.copy(A.T)
        src2 = np.copy(src)
        dst[:m, :] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        T_tot = np.identity(m+1)
        permutation_matrix = np.identity(N)

        for _ in range(max_iter):
            # find the nearest neighbors between the current source and destination points
            distances, indices = self.nearest_neighbor(
                src2[:m, :].T, dst[:m, :].T)

            # compute the transformation between the current source and nearest destination points
            T, _, _ = self.best_fit_transform(
                src2[:m, :].T, dst[:m, indices].T)
            permutation_matrix = permutation_matrix[indices]

            # update the current source
            T_tot = np.dot(T, T_tot)
            src2 = np.dot(T, src2)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        _, R, t = self.best_fit_transform(A, src2[:m, :].T)
        R_tot = T_tot[:3, :3].transpose()
        t_tot = T_tot[:3, 3]
        return R_tot, t_tot, distances

    def predict_single(self, src_pts, trgt_pts):
        """Predict a rotated version of src_pts that best match trgt_pts

        Args:
            src_pts (np.array): point cloud (dimension Nx3)
            trgt_pts (np.array): target point cloud (dimension Nx3)
        """
        # recover the rotation matrix
        R_tot, t_tot, _ = self.icp(src_pts, trgt_pts, tolerance=0.000000000001)
        pred_pts = src_pts.dot(R_tot) + t_tot
        pred_perm_matrix = IcpPredictor.deduce_permutation_matrix(

            pred_pts, trgt_pts)
        pred_pts = pred_perm_matrix.dot(pred_pts)
        return pred_pts

    def predict(self, src_pts, trgt_pts):
        """Predict a rotated version of src_pts that best match trgt_pts

        Args:
            src_pts (np.array): point cloud (dimension Nx3)
            trgt_pts (np.array): target point cloud (dimension Nx3)
        """
        predictions = []
        for i in range(self.N_rots):
            if i == 0:
                augm_pts = src_pts
            else:
                augm_rotation = generate_rotation_matrix()
                augm_pts = src_pts.dot(augm_rotation)
            pred_pts = self.predict_single(augm_pts, trgt_pts)
            pred2trgt_dist = np.mean((pred_pts-trgt_pts)**2)
            # we keep track of each prediction along with its associated distance
            predictions.append((pred_pts, pred2trgt_dist))

        best_pred = min(predictions, key=lambda x: x[1])
        return best_pred[0]
