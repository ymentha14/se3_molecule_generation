from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix


class RotationPredictor():
    @staticmethod
    def deduce_permutation_matrix(A, B):
        """Deduce the best permutation matrix transforming A into B
        using Hungarian algorithm, i.e return P such that WS(P @ A,B)
        is minimized
        """
        assert(A.shape == B.shape)
        N = A.shape[0]
        distances = distance_matrix(A, B)
        rows, cols = np.array(linear_sum_assignment(distances))
        P = np.zeros((N, N))
        P[cols, rows] = 1
        return P

    def __lt__(self, other):
        return self.max_iter < other.max_iter

    def __hash__(self):
        values = tuple(self.__dict__.values())
        return hash(values)


class DummyPredictor(RotationPredictor):
    def predict(self, src_pts, trgt_pts):
        pred_perm_matrix = DummyPredictor.deduce_permutation_matrix(
            src_pts, trgt_pts)
        pred_pts = pred_perm_matrix.dot(src_pts)
        return pred_pts


def WS(x, y):
    """
    Classic WS for numpy
    """
    return ((x-y)**2).mean()
