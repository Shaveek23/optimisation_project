import numpy as np
from tqdm import tqdm


def soft_thresholding(rho, lambda_):
    if rho < -lambda_:
        return rho + lambda_
    elif rho <= lambda_:
        return 0
    else:
        return rho - lambda_


class CoordinateDescent:

    def __init__(self, lambda_, intercept=True):

        self.y = None
        self.X = None
        self.theta = None
        self.path = None
        self.m = None
        self.n = None
        self.intercept = intercept
        self.lambda_ = lambda_

    def fit(self, X, y, iters=100, tol=1e-12):

        self.X = np.array(X.copy())
        self.y = np.array(y.copy())
        self.m, self.n = X.shape
        theta = np.zeros(self.n)
        path = theta.copy()

        for _ in tqdm(range(iters)):
            theta_new = theta.copy()
            for j in range(self.n):
                X_j = np.hstack((self.X[:, :j], self.X[:, (j + 1):]))
                theta_j = np.hstack((theta[:j], theta[(j + 1):])).reshape((-1, 1))

                rho_j = (self.X[:, j:(j + 1)] * (self.y - X_j @ theta_j)).sum()
                z_j = np.power(self.X[:, j], 2).sum()

                if self.intercept:
                    theta_new[j] = soft_thresholding(rho_j, self.lambda_) / z_j if j != 0 else rho_j
                else:
                    theta_new[j] = soft_thresholding(rho_j, self.lambda_) / z_j

                path = np.vstack((path, theta_new.copy()))

            if max(abs(theta_new - theta)) < tol:
                break
            else:
                theta = theta_new

        self.path = path
        self.theta = theta

    def predict(self, X):

        return X @ self.theta.reshape((self.n, 1))
