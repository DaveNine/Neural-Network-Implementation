import numpy as np
exp = np.exp

class ActivationFunction:

    class sigmoid:
        @staticmethod
        def sig(z):
            z = np.clip(z, -500, 500)
            return 1.0 / (1.0 + exp(-z))

        @staticmethod
        def sigprime(z):
            z = np.clip(z, -500, 500)
            return exp(-z) / (1.0 + exp(-z)) ** 2
