import numpy as np


class DisjointLinUCBDosingAlgorithm:
    def __init__(self, data, alpha=1):
        self.data = data
        self.alpha = alpha

    def learn_from_sample(self):
        na = 3
        actions = []
        d = self.data.shape[1] - 1
        p = np.zeros(na)
        theta = np.zeros((na, d))
        b = np.zeros((na, d))
        A = np.array([np.eye(d) for i in range(na)])
        for i in range(self.data.shape[0]):
            x = self.data.iloc[i, :-1].values
            optimal_action = self.data.iloc[i, -1]
            for action in range(na):
                Aa_inverse = np.linalg.inv(A[action])
                theta[action] = Aa_inverse.dot(b[action])
                p[action] = theta[action].dot(x) + self.alpha * np.sqrt(x.dot(Aa_inverse).dot(x))
            action = np.argmax(p)

            reward = 0 if action == optimal_action else -1
            A[action] = A[action] + np.outer(x, x.T)
            b[action] = b[action] + reward * x

            actions.append(action)
        return actions, self.data.Target.tolist()