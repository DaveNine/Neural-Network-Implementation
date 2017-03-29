import numpy as np
from ActivationFunction import ActivationFunction as AF


class CostFunction:
    class MSE:
        @staticmethod
        def f(a, y):
            return 0.5*np.linalg.norm(a-y)**2

        @staticmethod
        def delta(a, y, z):
            return (a-y) * AF.sigmoid().sigprime(z)

    class CrossEntropy:
        @staticmethod
        def f(a, y):
            return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

        @staticmethod
        def delta(a, y, z):
            return a - y