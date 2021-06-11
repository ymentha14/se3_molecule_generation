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

    # @staticmethod
    # def deduce_permutation_matrix(A, B):
    #     """deduce the best permutation matrix to transform matrix A
    #     into matrix B

    #     Args:
    #         A (np.array): matrix to transform
    #         B (np.array): target matrix

    #     Returns:
    #         np.array: permutation matrix
    #     """
    #     assert(A.shape == B.shape)
    #     N = A.shape[0]
    #     distances = distance_matrix(A, B)

    #     indexes = [np.unravel_index(ind, distances.shape)
    #                for ind in distances.flatten().argsort()]

    #     x_mask = [True for i in range(N)]
    #     y_mask = [True for i in range(N)]
    #     mapping = []
    #     for x, y in indexes:
    #         if x_mask[x] and y_mask[y]:
    #             mapping.append((x, y))
    #             x_mask[x] = False
    #             y_mask[y] = False

    #     mapping = [i[0] for i in sorted(mapping, key=lambda x: x[1])]

    #     comp_perm_matrix = np.identity(N)[mapping]
    #     return comp_perm_matrix


def WS(x, y):
    """
    Classic WS for numpy
    """
    return ((x-y)**2).mean()
