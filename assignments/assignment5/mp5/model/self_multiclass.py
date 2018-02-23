import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        for i in range(self.labels.shape[0]):
            temp = np.copy(y)
            temp[y != self.labels[i]] = 0
            temp[y == self.labels[i]] = 1
            clf = svm.LinearSVC(random_state=12345)
            binary_svm[self.labels[i]] = clf.fit(X, temp)
        return binary_svm

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        summation = np.hstack((np.reshape(y, (X.shape[0], 1)), X))
        for i in range(self.labels.shape[0]):
            for j in range(i + 1, self.labels.shape[0]):
                tempi = summation[y == i]
                tempj = summation[y == j]
                temp = np.vstack((tempi, tempj))
                clf = svm.LinearSVC(random_state=12345)
                binary_svm[tuple((i, j))] = clf.fit(temp[:, 1:], temp[:, 0])
        return binary_svm

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = []
        for idx in self.labels:
            scores.append(self.binary_svm[idx].decision_function(X))
        scores = np.array(scores)
        return np.transpose(scores)

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = []
        temp = []
        for idx in self.binary_svm:
            temp.append(self.binary_svm[idx].predict(X))

        np_temp = np.transpose(np.array(temp))

        for i in range(np_temp.shape[0]):
            item = np_temp[i, :]
            item = item.astype(np.int)
            scores.append(np.bincount(item, minlength=self.labels.shape[0]))
        return np.array(scores)

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        # Loss function = 1/2 * ||W||^2 + max(sum(1 - delta + w_j^Tx) - w_yi^Tx)
        # Loss of regularization
        reg_loss = 0.5 * np.trace(W.dot(W.T))

        # Number of observations and number of classes
        N = X.shape[0]
        K = W.shape[0]

        # Initialize Delta matrix
        Delta = np.zeros((K, N))
        for j in range(K):
            for i in range(N):
                if j == y[i]:
                    Delta[j, i] = 1
        I = np.ones((K, N))
        sub = I - Delta + W.dot(X.T)
        sub = np.reshape(np.amax(sub, axis=0), (N, 1))

        # Initialize w_yi^Tx
        foo = []
        for i in range(N):
            # foo[i] = W[y[i], :].dot((X[i, :]).T)
            foo.append(W[y[i]].dot((X[i]).T))
        foo = np.reshape(np.array(foo), (N, 1))

        # Max loss
        max_loss = C * np.sum(sub - foo)

        # Combine together
        total_loss = reg_loss + max_loss
        return total_loss

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        # Gradient of regularization
        reg_grad = W

        # Number of observations and number of classes
        N = X.shape[0]
        K = W.shape[0]

        # Initialize Delta matrix
        Delta = np.zeros((K, N))
        for j in range(K):
            for i in range(N):
                if j == y[i]:
                    Delta[j, i] = 1
        I = np.ones((K, N))
        sub = I - Delta + W.dot(X.T)
        # # reshape
        # rsub = np.reshape(np.amax(sub, axis=0), (N, 1))
        # Which j is the max in max expression
        idx = np.argmax(sub, axis=0)
        max_grad = np.zeros_like(W)

        # Compute gradient for max()
        for num, val in enumerate(idx):
            max_grad[val, :] += X[num, :]
            max_grad[y[num], :] -= X[num, :]

        # Combine together
        total_grad = reg_grad + max_grad * C
        return total_grad
