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
        self.costs = None
        self.intercept = intercept
        self.lambda_ = lambda_

    def fit(self, X, y, iters=1000, tol=1e-12, method = 'cyclic'):

        self.X = np.array(X.copy())
        self.y = np.array(y.copy())
        self.m, self.n = X.shape
        theta = np.zeros(self.n)
        path = theta.copy()


        cost = np.power(self.X @ np.zeros((self.n, 1)) - y, 2).sum()
        costs = [cost]

        i = 0
        last = 0

        for _ in tqdm(range(iters)):
            theta_new = theta.copy()

            if method in ('cyclic', 'randomized'):

                if method == 'cyclic':
                    j = (i % self.n)
                elif method == 'randomized':
                    j = np.random.choice(range(self.n))

                X_j = np.hstack((self.X[:, :j], self.X[:, (j + 1):]))
                theta_j = np.hstack((theta[:j], theta[(j + 1):])).reshape((-1, 1))

                rho_j = (self.X[:, j:(j + 1)] * (self.y - X_j @ theta_j)).sum()
                z_j = np.power(self.X[:, j], 2).sum()

                if self.intercept:
                    theta_new[j] = soft_thresholding(rho_j, self.lambda_) / z_j if j != 0 else rho_j
                else:
                    theta_new[j] = soft_thresholding(rho_j, self.lambda_) / z_j

                if abs(theta_new[j] - theta[j]) < tol:

                    last += 1

                    if (method == 'cyclic') and (last >= self.n):
                        break

                    if (method == 'randomized') and (last >= 2*self.n):
                        break

                else:
                    theta = theta_new
                    last = 0

            elif method == 'greedy':

                for j in range(self.n):

                    X_j = np.hstack((self.X[:, :j], self.X[:, (j + 1):]))
                    theta_j = np.hstack((theta[:j], theta[(j + 1):])).reshape((-1, 1))

                    rho_j = (self.X[:, j:(j + 1)] * (self.y - X_j @ theta_j)).sum()
                    z_j = np.power(self.X[:, j], 2).sum()

                    if self.intercept:
                        theta_new[j] = soft_thresholding(rho_j, self.lambda_) / z_j if j != 0 else rho_j
                    else:
                        theta_new[j] = soft_thresholding(rho_j, self.lambda_) / z_j

                diffs = np.abs(theta_new - theta)
                indx = np.where(diffs == diffs.max())

                if diffs.max() < tol:
                    break

                theta[indx] = theta_new[indx]

            path = np.vstack((path, theta_new.copy()))

            cost = np.power(self.X @ theta.reshape((self.n, 1)) - y, 2).sum()
            cost += self.lambda_*(np.abs(theta_new[1:]).sum()) if self.intercept else self.lambda_*(np.abs(theta_new).sum())
            costs.append(cost)

            i += 1

        self.costs = np.array(costs)
        self.path = path
        self.theta = theta

    def predict(self, X):

        return X @ self.theta.reshape((self.n, 1))
