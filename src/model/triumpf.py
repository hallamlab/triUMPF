'''
triUMPF: tri-Non-negative Matrix Factorization
'''

import copy
import logging
import os
import sys
import time
import warnings

import numpy as np
from scipy.sparse import lil_matrix, hstack
from scipy.special import expit
from scipy.stats import truncnorm
from sklearn import preprocessing
from sklearn.decomposition import randomized_svd
from sklearn.linear_model import SGDClassifier
from sklearn.utils._joblib import Parallel, delayed
from utility.access_file import save_data, load_data

logger = logging.getLogger(__name__)
EPSILON = np.finfo(np.float).eps
UPPER_BOUND = np.log(sys.float_info.max) * 10
LOWER_BOUND = np.log(sys.float_info.min) * 10
np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class triUMPF:
    def __init__(self, num_components=100, num_communities_p=100, num_communities_e=90, proxy_order_p=1,
                 proxy_order_e=1, mu_omega=1, mu_gamma=1, fit_features=False, fit_comm=False, fit_pure_comm = False,
                 normalize_input_feature=True, binarize_input_feature=False, use_external_features=True,
                 cutting_point=3650, fit_intercept=True, alpha=1e9, beta=1e9, rho=0.01,
                 lambdas=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01], eps=1e-4, early_stop=True, penalty='elasticnet',
                 alpha_elastic=0.0001, l1_ratio=0.65, loss_threshold=0.005, decision_threshold=0.35,
                 subsample_input_size=0.3, subsample_labels_size=50,
                 learning_type="optimal", lr=0.0001, lr0=0.0, delay_factor=1.0, forgetting_rate=0.9, batch=30,
                 max_inner_iter=5, num_epochs=200, num_jobs=-1, display_interval=5, shuffle=True,
                 random_state=12345, log_path='../../log'):

        logging.basicConfig(filename=os.path.join(log_path, 'triUMPF_events'), level=logging.DEBUG)
        np.random.seed(seed=random_state)
        self.num_components = num_components
        self.num_communities_p = num_communities_p
        self.num_communities_v = num_communities_e
        self.proxy_order_p = proxy_order_p
        self.proxy_order_e = proxy_order_e
        self.mu_omega = mu_omega
        self.mu_gamma = mu_gamma
        self.fit_features = fit_features
        self.fit_comm = fit_comm
        self.fit_pure_comm = fit_pure_comm
        self.normalize_input_feature = normalize_input_feature
        self.binarize_input_feature = binarize_input_feature
        if normalize_input_feature:
            self.binarize_input_feature = False
        self.use_external_features = use_external_features
        self.cutting_point = cutting_point
        self.fit_intercept = fit_intercept
        self.decision_threshold = decision_threshold
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.lambdas = lambdas
        self.lam_1 = lambdas[0]
        self.lam_2 = lambdas[1]
        self.lam_3 = lambdas[2]
        self.lam_4 = lambdas[3]
        self.lam_5 = lambdas[4]
        self.lam_6 = lambdas[5]
        self.penalty = penalty
        self.alpha_elastic = alpha_elastic
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.early_stop = early_stop
        self.loss_threshold = loss_threshold
        self.subsample_input_size = subsample_input_size
        self.subsample_labels_size = subsample_labels_size
        self.learning_type = learning_type
        self.lr = lr
        self.lr0 = lr0
        self.forgetting_rate = forgetting_rate
        self.delay_factor = delay_factor
        self.batch = batch
        self.max_inner_iter = max_inner_iter
        self.num_epochs = num_epochs
        self.num_jobs = num_jobs
        self.display_interval = display_interval
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = 0
        self.log_path = log_path
        warnings.filterwarnings("ignore", category=Warning)

    def __print_arguments(self, **kwargs):
        argdict = dict()
        argdict.update({'normalize_input_feature': 'Normalize data? {0}'.format(self.normalize_input_feature)})
        argdict.update({'binarize': 'Binarize data? {0}'.format(self.binarize_input_feature)})
        argdict.update(
            {'use_external_features': 'Whether to use external features? {0}'.format(self.use_external_features)})
        argdict.update(
            {'cutting_point': 'The cutting point after which preprocessing is halted: {0}'.format(self.cutting_point)})
        argdict.update({'num_components': 'Number of components: {0}'.format(self.num_components)})
        argdict.update({'fit_comm': 'Fit community structural information? {0}'.format(self.fit_comm)})
        argdict.update({'fit_pure_comm': 'Fit community excluding data? {0}'.format(self.fit_pure_comm)})
        argdict.update({'lambdas': 'Five hyper-parameters for constraints: {0}'.format(self.lambdas)})
        if self.fit_comm:
            argdict.update({'num_communities_p': 'Number of communities '
                                                 'for pathways: {0}'.format(self.num_communities_p)})
            argdict.update({'num_communities_v': 'Number of communities '
                                                 'for ecs: {0}'.format(self.num_communities_v)})
            argdict.update({'alpha': 'A hyper-parameter (orthogonal condition): {0}'.format(self.alpha)})
            argdict.update({'beta': 'A hyper-parameter (orthogonal condition): {0}'.format(self.beta)})
            argdict.update(
                {'rho': 'A hyper-parameter to fuse coefficients with association matrix.: {0}'.format(self.rho)})
            argdict.update({'penalty': 'The penalty (aka regularization term): {0}'.format(self.penalty)})
            if self.penalty == "elasticnet":
                argdict.update({'alpha-elastic': 'Constant controlling elastic term: {0}'.format(self.alpha_elastic)})
                argdict.update({'l1_ratio': 'The elastic net mixing parameter: {0}'.format(self.l1_ratio)})
                argdict.update({'--alpha-elastic': 'Constant controlling elastic term: {0}'.format(self.alpha_elastic)})
            argdict.update({'fit_intercept': 'Whether the intercept should be estimated '
                                             'or not? {0}'.format(self.fit_intercept)})
        argdict.update({'eps': 'Truncate all values less then this in output to zero: {0}'.format(self.eps)})
        argdict.update(
            {'loss_threshold': 'A cutoff threshold between two consecutive rounds: {0}'.format(self.loss_threshold)})
        argdict.update({'early_stop': 'Whether to apply early stopping criteria? {0}'.format(self.early_stop)})
        argdict.update({'decision_threshold': 'The decision cutoff threshold: {0}'.format(self.decision_threshold)})
        argdict.update({'subsample_input_size': 'Subsampling inputs: {0}'.format(self.subsample_input_size)})
        argdict.update({'subsample_labels_size': 'Subsampling labels: {0}'.format(self.subsample_labels_size)})
        argdict.update({'learning_type': 'The learning rate schedule: {0}'.format(self.learning_type)})
        if self.learning_type == "optimal":
            argdict.update({'lr': 'The learning rate: {0}'.format(self.lr)})
            argdict.update({'lr0': 'The initial learning rate: {0}'.format(self.lr0)})
        else:
            argdict.update({'forgetting_rate': 'Forgetting rate to control how quickly old '
                                               'information is forgotten: {0}'.format(self.forgetting_rate)})
            argdict.update({'delay_factor': 'Delay factor down weights '
                                            'early iterations: {0}'.format(self.delay_factor)})
        argdict.update({'batch': 'Number of examples to use in each iteration: {0}'.format(self.batch)})
        argdict.update({'max_inner_iter': 'Number of inner loops inside an optimizer: {0}'.format(self.max_inner_iter)})
        argdict.update({'num_epochs': 'Number of loops over training set: {0}'.format(self.num_epochs)})
        argdict.update({'num_jobs': 'Number of parallel workers: {0}'.format(self.num_jobs)})
        argdict.update({'display_interval': 'How often to evaluate? {0}'.format(self.display_interval)})
        argdict.update({'shuffle': 'Shuffle the dataset? {0}'.format(self.shuffle)})
        argdict.update({'random_state': 'The random number generator: {0}'.format(self.random_state)})
        argdict.update({'log_path': 'Logs are stored in: {0}'.format(self.log_path)})

        for key, value in kwargs.items():
            argdict.update({key: value})
        args = list()
        for key, value in argdict.items():
            args.append(value)
        args = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(args))), args)]
        args = '\n\t\t'.join(args)
        print('\t>> The following arguments are applied:\n\t\t{0}'.format(args), file=sys.stderr)
        logger.info('\t>> The following arguments are applied:\n\t\t{0}'.format(args))

    def __init_variables(self, M, num_components, init="nndsvd"):
        """Algorithms for NMF initialization. Similar implemntation as the
        one described in sklearn.decomposition module
        """

        if init != "nndsvd" or init is None:
            bound = np.sqrt(M.mean() / num_components)
            W = truncnorm.rvs(-bound, bound, size=(M.shape[0], num_components))
            H = truncnorm.rvs(-bound, bound, size=(num_components, M.shape[1]))
        else:
            # NNDSVD initialization
            U, S, V = randomized_svd(M, num_components, random_state=self.random_state)
            W, H = np.zeros(U.shape), np.zeros(V.shape)
            # The leading singular triplet is non-negative
            # so it can be used as is for initialization.
            W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
            H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])
            if num_components > M.shape[1]:
                num_components = U.shape[1]
            for j in np.arange(start=0, stop=num_components):
                u, v = U[:, j], V[j, :]
                # extract positive and negative parts of column vectors
                u_pos = np.maximum(u, 0)
                v_pos = np.maximum(v, 0)
                u_neg = np.abs(np.minimum(u, 0))
                v_neg = np.abs(np.minimum(v, 0))

                # and their norms
                u_pos_norm = np.linalg.norm(u_pos)
                v_pos_norm = np.linalg.norm(v_pos)
                u_neg_norm = np.linalg.norm(u_neg)
                v_neg_norm = np.linalg.norm(v_neg)
                sim_pos = u_pos_norm * v_pos_norm
                sim_neg = u_neg_norm * v_neg_norm

                # choose update
                if sim_pos > sim_neg:
                    u = u_pos / u_pos_norm
                    v = v_pos / v_pos_norm
                    sigma = sim_pos
                else:
                    u = u_neg / u_neg_norm
                    v = v_neg / v_neg_norm
                    sigma = sim_neg
                lbd = np.sqrt(S[j] * sigma)
                W[:, j] = lbd * u
                H[j, :] = lbd * v
        W[W < self.eps] = 0.
        H[H < self.eps] = 0.
        np.nan_to_num(W, copy=False)
        np.nan_to_num(H, copy=False)
        return W, H

    def __shffule(self, num_samples):
        if self.shuffle:
            idx = np.arange(num_samples)
            np.random.shuffle(idx)
            return idx

    def __check_bounds(self, X):
        X = np.clip(X, LOWER_BOUND, UPPER_BOUND)
        if len(X.shape) > 1:
            if X.shape[0] == X.shape[1]:
                min_x = np.min(X) + EPSILON
                max_x = np.max(X) + EPSILON
                X = X - min_x
                X = X / (max_x - min_x)
                X = 2 * X - 1
        return X

    def __solver(self, X, y, coef, intercept):
        """Initialize logistic regression variables."""
        penalty = "elasticnet"
        if self.penalty != "elasticnet":
            penalty = "none"
        estimator = SGDClassifier(loss='log', penalty=penalty, alpha=self.alpha_elastic,
                                  l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept,
                                  max_iter=self.max_inner_iter, shuffle=self.shuffle,
                                  n_jobs=self.num_jobs, random_state=self.random_state,
                                  warm_start=True, average=True)
        estimator.fit(X=X, y=y, coef_init=coef, intercept_init=intercept)
        return estimator.coef_[0], estimator.intercept_

    def __optimal_learning_rate(self, alpha):
        def _loss(p, y):
            z = p * y
            # approximately equal and saves the computation of the log
            if z > 18:
                return np.exp(-z)
            if z < -18:
                return -z
            return np.log(1.0 + np.exp(-z))

        typw = np.sqrt(1.0 / np.sqrt(alpha))
        # computing lr0, the initial learning rate
        initial_eta0 = typw / max(1.0, _loss(-typw, 1.0))
        # initialize t such that lr at first sample equals lr0
        optimal_init = 1.0 / (initial_eta0 * alpha)
        return optimal_init

    def __sigmoid(self, X):
        return expit(X)

    def __log_logistic(self, X, negative=True):
        param = 1
        if negative:
            param = -1
        X = np.clip(X, EPSILON, 1 - EPSILON)
        X = param * np.log(1 + np.exp(X))
        return X

    def __grad_l21_norm(self, M):
        if len(M.shape) == 2:
            D = 1 / (2 * np.linalg.norm(M, axis=1))
            ret = np.dot(np.diag(D), M)
        else:
            D = (2 * np.linalg.norm(M) + EPSILON)
            ret = M / D
        return ret

    def __reconstruction_error(self, M, X1, X2=None):
        X12 = X1
        if X2 is not None:
            X12 = np.dot(X1, X2)
        rec_err = np.square(np.linalg.norm((M - X12)))
        rec_err = np.sqrt(2 * rec_err)
        return rec_err

    def rec_err(self, M, M_hat):
        assert M.shape == M_hat.shape
        err = self.__reconstruction_error(M, X1=M_hat)
        return err

    def __cost_logistic(self, X, y, label_idx):
        coef_intercept_label = self.coef_label[label_idx].toarray()
        if self.fit_intercept:
            coef_intercept_label = np.hstack((self.intercept_label[label_idx].toarray(), coef_intercept_label))
        cond = -(2 * y[:, label_idx] - 1)
        coef = np.dot(X, coef_intercept_label.T)
        coef = np.multiply(coef, cond)
        res = -np.mean(self.__log_logistic(coef))
        return res

    def __cost(self, M, X=None, y=None, P=None, E=None, A=None, B=None):
        loss = 0.0
        if M is not None:
            loss += self.__reconstruction_error(M=M.toarray(), X1=self.W.toarray(), X2=self.H.toarray().T)
            if P is not None and E is not None:
                loss += self.__reconstruction_error(M=self.W.toarray(), X1=P.toarray(), X2=self.U.toarray())
                loss += self.__reconstruction_error(M=self.H.toarray(), X1=E.toarray(), X2=self.V.toarray())
                loss += self.__reconstruction_error(M=self.U.toarray(), X1=self.V.toarray())
        else:
            RK_t = np.dot(self.R.toarray(), self.K.toarray().T)
            TC_t = np.dot(self.T.toarray(), self.C.toarray().T)
            loss += self.__reconstruction_error(M=A.toarray(), X1=P.toarray(), X2=TC_t)
            loss += self.__reconstruction_error(M=B.toarray(), X1=E.toarray(), X2=RK_t)
            if not self.fit_pure_comm:
                loss += self.__reconstruction_error(M=X[:, :self.cutting_point].toarray(),
                                                    X1=self.L.toarray(), X2=RK_t)
                loss += self.__reconstruction_error(M=y.toarray(), X1=self.L.toarray(), X2=TC_t)
            if self.fit_intercept:
                X = hstack((lil_matrix(np.ones((X.shape[0], 1))), X))
            X = X.toarray()
            y = y.toarray()
            labels = np.arange(self.num_labels)
            if labels > self.subsample_labels_size:
                labels = np.random.choice(np.arange(self.num_labels), self.subsample_labels_size, replace=False)
            parallel = Parallel(n_jobs=self.num_jobs, verbose=max(0, self.verbose - 1))
            results = parallel(delayed(self.__cost_logistic)(X, y, label_idx)
                               for label_idx in labels)
            loss += np.mean(results)
        return loss

    def __optimize_w(self, M, Q, P, current_progress, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...{2}'.format("W",
                                                                 ((current_progress / total_progress) * 100),
                                                                 " " * 10)
        print(desc, end="\r")

        numerator = M.dot(self.H)
        if P is not None:
            numerator += self.lam_1 * P.dot(self.U)
        numerator[numerator < self.eps] = 0.
        denominator = self.W.dot(self.H.T.dot(self.H).toarray() + Q)
        denominator[denominator == 0] = EPSILON
        tmp = (numerator / denominator)
        self.W = self.W.multiply(tmp)
        self.W = lil_matrix(self.W)

    def __optimize_h(self, M, Q, E, current_progress, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...{2}'.format("H",
                                                                 ((current_progress / total_progress) * 100),
                                                                 " " * 10)
        print(desc, end="\r")

        numerator = M.T.dot(self.W)
        if E is not None:
            numerator += self.lam_1 * E.dot(self.V)
        numerator[numerator < self.eps] = 0.
        denominator = self.H.dot(self.W.T.dot(self.W).toarray() + Q)
        denominator[denominator == 0] = EPSILON
        tmp = (numerator / denominator)
        self.H = self.H.multiply(tmp)
        self.H = lil_matrix(self.H)

    def __optimize_u(self, P, D, current_progress, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...{2}'.format("U",
                                                                 ((current_progress / total_progress) * 100),
                                                                 " " * 10)
        print(desc, end="\r")

        numerator = self.lam_1 * P.T.dot(self.W) + self.lam_3 * self.V
        numerator[numerator < self.eps] = 0.
        denominator = self.lam_1 * P.T.dot(P).dot(self.U)
        denominator = denominator + self.U * D
        denominator[denominator == 0] = EPSILON
        tmp = (numerator / denominator)
        self.U = self.U.multiply(tmp)
        self.U = lil_matrix(self.U)

    def __optimize_v(self, E, D, current_progress, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...{2}'.format("V",
                                                                 ((current_progress / total_progress) * 100),
                                                                 " " * 10)
        print(desc, end="\r")

        numerator = self.lam_2 * E.T.dot(self.H) + self.lam_3 * self.U
        numerator[numerator < self.eps] = 0.
        denominator = self.lam_2 * E.T.dot(E).dot(self.V)
        denominator = denominator + self.V * D
        denominator[denominator == 0] = EPSILON
        tmp = (numerator / denominator)
        self.V = self.V.multiply(tmp)
        self.V = lil_matrix(self.V)

    def __optimize_c(self, y, A, P, samples_idx, current_progress, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...{2}'.format("C",
                                                                 ((current_progress / total_progress) * 100),
                                                                 " " * 10)
        print(desc, end="\r")

        numerator = A.T.dot(P).dot(self.T)
        if not self.fit_pure_comm:
            numerator += y[samples_idx].T.dot(self.L[samples_idx]).dot(self.T)
        numerator += 2 * self.alpha * self.C
        numerator[numerator < self.eps] = 0.
        denominator = self.T.T.dot(P.T).dot(P).dot(self.T)
        if not self.fit_pure_comm:
            denominator += self.T.T.dot(self.L[samples_idx].T).dot(self.L[samples_idx]).dot(self.T)
        denominator += 2 * self.alpha * self.C.T.dot(self.C).toarray() + self.lam_5
        denominator = self.C.dot(lil_matrix(denominator))
        denominator[denominator == 0] = EPSILON
        tmp = (numerator / denominator)
        self.C = self.C.multiply(tmp)
        self.C = lil_matrix(self.C)

    def __optimize_k(self, X, B, E, samples_idx, current_progress, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...{2}'.format("K",
                                                                 ((current_progress / total_progress) * 100),
                                                                 " " * 10)
        print(desc, end="\r")

        numerator = B.T.dot(E).dot(self.R)
        if not self.fit_pure_comm:
            numerator += X[samples_idx].T.dot(self.L[samples_idx]).dot(self.R)
        numerator += 2 * self.beta * self.K
        numerator[numerator < self.eps] = 0.
        denominator = self.R.T.dot(E.T).dot(E).dot(self.R)
        if not self.fit_pure_comm:
            denominator += self.R.T.dot(self.L[samples_idx].T).dot(self.L[samples_idx]).dot(self.R.toarray())
        denominator += (2 * self.beta * self.K.T.dot(self.K)).toarray() + self.lam_5
        denominator = self.K.dot(lil_matrix(denominator))
        denominator[denominator == 0] = EPSILON
        tmp = (numerator / denominator)
        self.K = self.K.multiply(tmp)
        self.K = lil_matrix(self.K)

    def __optimize_t(self, y, A, P, samples_idx, batch_idx, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...{2}'.format("T",
                                                                 ((batch_idx / total_progress) * 100),
                                                                 " " * 10)
        print(desc, end="\r")

        numerator = P.T.dot(A).dot(self.C)
        if not self.fit_pure_comm:
            numerator += self.L[samples_idx].T.dot(y[samples_idx]).dot(self.C)
        numerator[numerator < self.eps] = 0.
        denominator = P.T.dot(self.C).dot(self.C.T).dot(P).toarray() + self.lam_5
        denominator = lil_matrix(denominator).dot(self.T)
        if not self.fit_pure_comm:
            tmp = self.L[samples_idx].T.dot(self.L[samples_idx]).dot(self.T)
            tmp = tmp.dot(self.C.T).dot(self.C)
            denominator += tmp
        denominator[denominator == 0] = EPSILON
        tmp = (numerator / denominator)
        self.T = self.T.multiply(tmp)
        self.T = lil_matrix(self.T)

    def __optimize_r(self, X, B, E, samples_idx, batch_idx, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...{2}'.format("R",
                                                                 ((batch_idx / total_progress) * 100),
                                                                 " " * 10)
        print(desc, end="\r")

        numerator = E.T.dot(B).dot(self.K)
        if not self.fit_pure_comm:
            numerator += self.L[samples_idx].T.dot(X[samples_idx]).dot(self.K)
        numerator[numerator < self.eps] = 0.
        denominator = E.T.dot(self.K).dot(self.K.T).dot(E).toarray() + self.lam_5
        denominator = lil_matrix(denominator).dot(self.R)
        if not self.fit_pure_comm:
            tmp = self.L[samples_idx].T.dot(self.L[samples_idx]).dot(self.R).dot(self.K.T).dot(self.K)
            denominator += tmp
        denominator[denominator == 0] = EPSILON
        tmp = (numerator / denominator)
        self.R = np.multiply(self.R.toarray(), tmp)
        self.R = lil_matrix(self.R)

    def __optimize_theta_label(self, X, y, W, H, samples_idx, learning_rate, batch_idx, total_progress):
        X = X[samples_idx].toarray()
        y = y[samples_idx].toarray()
        num_samples = X.shape[0]
        count = batch_idx * self.num_labels

        # pre-calculate and store this value for regularization purposes
        ZHWt_t = self.Z.dot(H).dot(W.T).T

        for label_idx in np.arange(self.num_labels):
            desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...{2}'.format("Theta",
                                                                     ((batch_idx + count) / total_progress) * 100,
                                                                     " " * 10)
            print(desc, end="\r")
            count += 1
            gradient = 0.0

            ## If only positive or negative instances then return the function
            if len(np.unique(y[:, label_idx])) < 2:
                if np.sum(self.coef_label[label_idx].toarray()) != 0.0:
                    continue
                if np.unique(y[:, label_idx]) == 1:
                    self.coef_label[label_idx] = lil_matrix(np.ones(self.coef_label[label_idx].shape[0]))
                else:
                    self.coef_label[label_idx] = lil_matrix(-np.ones(self.coef_label[label_idx].shape[0]))
                continue
            coef = np.reshape(self.coef_label[label_idx].toarray(), newshape=(1, self.coef_label[label_idx].shape[1]))
            intercept = self.intercept_label[label_idx].toarray()[0]
            coef, intercept = self.__solver(X=X, y=y[:, label_idx], coef=coef, intercept=intercept)
            self.coef_label[label_idx] = lil_matrix(coef)
            self.intercept_label[label_idx] = lil_matrix(intercept)

            gradient += 2 * self.rho * (self.coef_label[label_idx, :self.cutting_point] - ZHWt_t[label_idx])
            if self.penalty == "l21":
                gradient += self.lam_6 * self.__grad_l21_norm(
                    M=self.coef_label[label_idx, :self.cutting_point].toarray())
            self.coef_label[label_idx, :self.cutting_point] = self.coef_label[label_idx,
                                                              :self.cutting_point] - learning_rate * gradient
            gradient = 0.0
            if (self.coef_label[label_idx, self.cutting_point:].shape[1]) > 0:
                if self.penalty == "l21":
                    gradient = self.lam_6 * self.__grad_l21_norm(
                        M=self.coef_label[label_idx, self.cutting_point:].toarray())
                self.coef_label[label_idx, self.cutting_point:] = self.coef_label[label_idx,
                                                                  self.cutting_point:] - learning_rate * gradient
        self.coef_label = lil_matrix(self.coef_label)

    def __optimize_l(self, X, y, samples_idx, learning_rate, batch_idx, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...{2}'.format("L",
                                                                 ((batch_idx / total_progress) * 100),
                                                                 " " * 10)
        print(desc, end="\r")

        CT_t = self.C.dot(self.T.T)
        KR_t = self.K.dot(self.R.T)
        gradient = self.L[samples_idx].dot(self.T).dot(self.C.T).dot(CT_t)
        gradient += self.L[samples_idx].dot(self.R).dot(self.K.T).dot(KR_t)
        gradient -= y[samples_idx].dot(CT_t)
        gradient -= X[samples_idx].dot(KR_t)
        gradient += self.lam_6 * self.L[samples_idx]
        gradient = preprocessing.minmax_scale(X=gradient.toarray(), feature_range=(-1, 1))
        self.L[samples_idx] = self.L[samples_idx] - learning_rate * 2 * lil_matrix(gradient)
        self.L = lil_matrix(self.L)

    def __optimize_z(self, W, H, learning_rate):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...{2}'.format("Z", 100, " " * 10)
        print(desc, end="\r")

        WH_t = W.dot(H.T)
        gradient = self.rho * self.Z.dot(H).dot(W.T).dot(WH_t)
        gradient -= self.rho * self.coef_label[:, :self.cutting_point].T.dot(WH_t)
        gradient += self.lam_6 * self.Z
        gradient = preprocessing.minmax_scale(X=gradient.toarray(), feature_range=(-1, 1))
        self.Z = self.Z - learning_rate * 2 * lil_matrix(gradient)
        self.Z = lil_matrix(self.Z)

    def __optimize_fact(self, M, P, E, current_progress=None, total_progress=None):
        """Compute Non-negative Matrix Factorization with Multiplicative Update"""
        Q = self.lam_1 + self.lam_4
        self.__optimize_w(M=M, Q=Q, P=P, current_progress=current_progress,
                          total_progress=total_progress)
        self.__optimize_h(M=M, Q=Q, E=E, current_progress=current_progress,
                          total_progress=total_progress)
        if P is not None or E is not None:
            D = self.lam_3 + self.lam_4
            self.__optimize_u(P=P, D=D, current_progress=current_progress,
                              total_progress=total_progress)
            self.__optimize_v(E=E, D=D, current_progress=current_progress,
                              total_progress=total_progress)

    def __optimize_comm(self, X, y, P, E, A, B, samples_idx):
        """Compute Non-negative Matrix Factorization with Multiplicative Update"""
        parallel = Parallel(n_jobs=1, verbose=max(0, self.verbose - 1))
        list_batches = np.arange(start=0, stop=len(samples_idx), step=self.batch)
        current_progress = 1
        total_progress = len(list_batches) * self.max_inner_iter * 2
        for iter in np.arange(start=1, stop=self.max_inner_iter + 1):
            # optimize C
            parallel(delayed(self.__optimize_c)(y, A, P, samples_idx[batch:batch + self.batch],
                                                current_progress + batch_idx, total_progress)
                     for batch_idx, batch in enumerate(list_batches))
            current_progress += len(list_batches)
            # optimize K
            parallel(delayed(self.__optimize_k)(X, B, E, samples_idx[batch:batch + self.batch],
                                                current_progress + batch_idx, total_progress)
                     for batch_idx, batch in enumerate(list_batches))
            current_progress += len(list_batches)

    def __optimize_path(self, X, y, W, H, P, E, A, B, samples_idx, learning_rate):
        parallel = Parallel(n_jobs=1, verbose=max(0, self.verbose - 1))
        list_batches = np.arange(start=0, stop=len(samples_idx), step=self.batch)

        # optimize T
        parallel(delayed(self.__optimize_t)(y, A, P, samples_idx[batch:batch + self.batch],
                                            batch_idx, len(list_batches))
                 for batch_idx, batch in enumerate(list_batches))
        # optimize R
        parallel(delayed(self.__optimize_r)(X[:, :self.cutting_point], B, E, samples_idx[batch:batch + self.batch],
                                            batch_idx, len(list_batches))
                 for batch_idx, batch in enumerate(list_batches))
        # optimize Theta
        total_progress = len(list_batches) * self.num_labels
        parallel(delayed(self.__optimize_theta_label)(X, y, W, H, samples_idx[batch:batch + self.batch],
                                                      learning_rate, batch_idx, total_progress)
                 for batch_idx, batch in enumerate(list_batches))
        # optimize L
        if not self.fit_pure_comm:
            parallel(delayed(self.__optimize_l)(X[:, :self.cutting_point], y, samples_idx[batch:batch + self.batch],
                                                learning_rate, batch_idx, len(list_batches))
                    for batch_idx, batch in enumerate(list_batches))
        # optimize Z
        self.__optimize_z(W=W, H=H, learning_rate=learning_rate)

    def fit(self, M=None, W=None, H=None, X=None, y=None, P=None, E=None, A=None, B=None,
            model_name='triUMPF', model_path="../../model", result_path=".",
            display_params: bool = True):

        """Learn a NMF model for the data X.
        Parameters
        ----------
        M : {array-like, sparse matrix}, shape (n_pathways, n_ecs)
            Data matrix to be decomposed
        """
        if W is None or H is None:
            if M is None:
                raise Exception("Please provide an association matrix.")
        else:
            assert W.shape[1] == H.shape[1]
        if P is not None:
            if M is not None:
                assert M.shape[0] == P.shape[0]
            else:
                assert W.shape[0] == P.shape[0]
            assert P.shape[0] >= self.num_components
        if E is not None:
            if M is not None:
                assert M.shape[1] == E.shape[0]
            else:
                assert H.shape[0] == E.shape[0]
            assert E.shape[0] >= self.num_components
        if self.fit_comm:
            if X is None or y is None:
                raise Exception("Please provide a dataset.")
            if A is None or B is None or P is None or E is None:
                raise Exception("Please provide two adjacency matrices and two feature matrices.")
            assert X.shape[0] == y.shape[0]
            if M is not None:
                assert M.shape[0] == A.shape[0]
                assert M.shape[1] == B.shape[0]
            else:
                assert W.shape[0] == A.shape[0]
                assert H.shape[0] == B.shape[0]

        #######################################################################################################
        ###################################            Initialize           ###################################
        #######################################################################################################

        if W is None and H is None:
            # initialize W and H
            self.W, self.H = self.__init_variables(M=M, num_components=self.num_components)
            self.W = lil_matrix(self.W)
            self.H = lil_matrix(self.H.T)
            if P is not None and E is not None:
                _, self.U = self.__init_variables(M=P, num_components=self.num_components)
                _, self.V = self.__init_variables(M=E, num_components=self.num_components)
                self.U = lil_matrix(self.U.T)
                self.V = lil_matrix(self.V.T)

        if self.fit_comm:
            # binarize or normalize features
            if self.binarize_input_feature:
                if self.use_external_features:
                    X[:, :self.cutting_point] = preprocessing.binarize(X[:, :self.cutting_point])
                else:
                    X = preprocessing.binarize(X)
            if self.normalize_input_feature:
                if self.use_external_features:
                    X[:, :self.cutting_point] = preprocessing.normalize(X[:, :self.cutting_point])
                else:
                    X = preprocessing.normalize(X)
            X = lil_matrix(X)
            self.num_labels = y.shape[1]

            # compute proxy
            tmp_order_p = [i for i in np.arange(start=1, stop=self.proxy_order_p + 1)]
            tmp_order_e = [i for i in np.arange(start=1, stop=self.proxy_order_e + 1)]
            A = lil_matrix(np.sum([(self.mu_omega ** (i - 1)) * A.power(i) for i in tmp_order_p]))
            B = lil_matrix(np.sum([(self.mu_gamma ** (i - 1)) * B.power(i) for i in tmp_order_e]))

            # intialize parameters
            self.C, _ = self.__init_variables(M=P, num_components=self.num_communities_p, init=None)
            self.K, _ = self.__init_variables(M=E, num_components=self.num_communities_v, init=None)
            _, self.T = self.__init_variables(M=P, num_components=self.num_communities_p, init=None)
            _, self.R = self.__init_variables(M=E, num_components=self.num_communities_v, init=None)
            self.C = lil_matrix(self.C)
            self.K = lil_matrix(self.K)
            self.T = lil_matrix(self.T.T)
            self.R = lil_matrix(self.R.T)
            if not self.fit_pure_comm:
                self.L = np.random.gamma(shape=self.T.shape[0], scale=1 / self.T.shape[0],
                                         size=(X.shape[0], self.T.shape[0]))
            tmp = X[:, :self.cutting_point].shape[1]
            if H is not None:
                self.Z = np.random.gamma(shape=H.shape[1], scale=1 / H.shape[1], size=(tmp, H.shape[0]))
            else:
                self.Z = np.random.gamma(shape=self.H.shape[1], scale=1 / self.H.shape[1],
                                         size=(tmp, self.H.shape[0]))
            self.Z = lil_matrix(self.Z)
            if not self.fit_pure_comm:
                self.L = lil_matrix(self.L)
            self.coef_label = lil_matrix(np.zeros(shape=(self.num_labels, X.shape[1])))
            self.intercept_label = lil_matrix(np.zeros(shape=(self.num_labels, 1)))

        if display_params:
            if self.fit_comm:
                self.__print_arguments(num_labels='Number of labels: {0}'.format(self.num_labels))
            else:
                self.__print_arguments()
            time.sleep(2)

        print('\t>> Training triUMPF...')
        logger.info('\t>> Training triUMPF...')
        timeref = time.time()
        n_epochs = self.num_epochs + 1

        cost_file_name = model_name + "_cost.txt"
        save_data('', file_name=cost_file_name, save_path=result_path, mode='w', w_string=True, print_tag=False)

        #######################################################################################################
        ###################################           First Phase           ###################################
        #######################################################################################################

        if W is None and H is None:
            desc = '\t\t>>> Decomposing M matrix...'
            print(desc)
            logger.info(desc)

            data = "Decomposition Cost: \n"
            save_data(data, file_name=cost_file_name, save_path=result_path, mode='a', w_string=True, print_tag=False)

            # record progress
            old_cost = np.inf
            for epoch in np.arange(start=1, stop=n_epochs):
                # set epoch time
                start_epoch = time.time()

                # optimize
                self.__optimize_fact(M=M, P=P, E=E, current_progress=epoch, total_progress=self.num_epochs)

                end_epoch = time.time()
                self.is_fit = True

                # Save models parameters based on test frequencies
                if (epoch % self.display_interval) == 0 or epoch == 1 or epoch == n_epochs - 1:
                    new_cost = self.__cost(M=M, X=X, y=y, P=P, E=E, A=A, B=B) / self.num_epochs
                    data = str(epoch) + '\t' + str(round(end_epoch - start_epoch, 3)) + '\t' + str(
                        new_cost) + '\t' + str(old_cost) + '\n'
                    save_data(data=data, file_name=cost_file_name, save_path=result_path, mode='a', w_string=True,
                              print_tag=False)
                    logger.info(data)
                    if old_cost >= new_cost or epoch == n_epochs - 1:
                        W_name = model_name + "_W.pkl"
                        H_name = model_name + "_H.pkl"
                        U_name = model_name + "_U.pkl"
                        V_name = model_name + "_V.pkl"

                        if old_cost >= new_cost:
                            logger.info('\t\t  >> Storing W to: {0}'.format(W_name))
                            save_data(data=self.W, file_name=W_name, save_path=model_path, mode="wb", print_tag=False)
                            logger.info('\t\t  >> Storing H to: {0}'.format(H_name))
                            save_data(data=self.H, file_name=H_name, save_path=model_path, mode="wb", print_tag=False)
                        if epoch == n_epochs - 1:
                            W_name = model_name + '_W_final.pkl'
                            H_name = model_name + '_H_final.pkl'
                            logger.info('\t\t  >> Storing W to: {0}'.format(W_name))
                            save_data(data=self.W, file_name=W_name, save_path=model_path, mode="wb", print_tag=False)
                            logger.info('\t\t  >> Storing H to: {0}'.format(H_name))
                            save_data(data=self.H, file_name=H_name, save_path=model_path, mode="wb", print_tag=False)

                        if P is not None and E is not None:
                            if old_cost >= new_cost:
                                logger.info('\t\t  >> Storing U to: {0}'.format(U_name))
                                save_data(data=self.U, file_name=U_name, save_path=model_path, mode="wb",
                                          print_tag=False)
                                logger.info('\t\t  >> Storing V to: {0}'.format(V_name))
                                save_data(data=self.V, file_name=V_name, save_path=model_path, mode="wb",
                                          print_tag=False)
                            if epoch == n_epochs - 1:
                                U_name = model_name + "_U.pkl"
                                V_name = model_name + "_V.pkl"
                                logger.info('\t\t  >> Storing U to: {0}'.format(U_name))
                                save_data(data=self.U, file_name=U_name, save_path=model_path, mode="wb",
                                          print_tag=False)
                                logger.info('\t\t  >> Storing V to: {0}'.format(V_name))
                                save_data(data=self.V, file_name=V_name, save_path=model_path, mode="wb",
                                          print_tag=False)

                        desc = '\t\t           --> Storing the triUMPF model to: {0:s}.pkl'.format(model_name)
                        logger.info(desc)
                        W = self.W
                        H = self.H
                        del self.W, self.H
                        if P is not None and E is not None:
                            del self.U, self.V
                        save_data(data=copy.copy(self), file_name=model_name + '.pkl', save_path=model_path, mode="wb",
                                  print_tag=False)

                        if epoch != n_epochs - 1:
                            self.W = W
                            self.H = H
                            if P is not None and E is not None:
                                self.U = load_data(file_name=U_name, load_path=model_path,
                                                   tag="triUMPF\'s U parameters", print_tag=False)
                                self.V = load_data(file_name=V_name, load_path=model_path,
                                                   tag="triUMPF\'s V parameters", print_tag=False)
                        if self.early_stop:
                            relative_change = np.abs((new_cost - old_cost) / old_cost)
                            desc = '\t\t  --> There is a little improvement in the reconstruction error '
                            desc += '(< {0}) for epoch {1}, hence, training is terminated...'.format(
                                self.loss_threshold,
                                epoch)
                            if relative_change < self.loss_threshold:
                                logger.info(desc)
                                break
                        old_cost = new_cost

        #######################################################################################################
        ##################################            Second Phase           ##################################
        #######################################################################################################

        if self.fit_comm:
            desc = '\t\t>>> Learning Community and Pathway Parameters...'
            print(desc)
            logger.info(desc)

            data = "Community Cost: \n"
            save_data(data, file_name=cost_file_name, save_path=result_path, mode='a', w_string=True, print_tag=False)

            if self.learning_type == "optimal":
                optimal_init = self.__optimal_learning_rate(alpha=self.lr)

            # record progress
            old_cost = np.inf
            for epoch in np.arange(start=1, stop=n_epochs):
                desc = '\t\t   {0:d})- Epoch count ({0:d}/{1:d})...{2}'.format(epoch, n_epochs - 1, " " * 20)
                print(desc)
                logger.info(desc)

                # shuffle dataset
                sample_idx = self.__shffule(num_samples=X.shape[0])
                X = X[sample_idx, :]
                y = y[sample_idx, :]
                if not self.fit_pure_comm:
                    self.L = self.L[sample_idx, :]

                if self.learning_type == "optimal":
                    # usual optimization technique
                    learning_rate = 1.0 / (self.lr * (optimal_init + epoch - 1))
                else:
                    # using variational inference sgd
                    learning_rate = np.power((epoch + self.delay_factor), -self.forgetting_rate)

                samples = np.arange(X.shape[0])
                size_x = int(np.ceil(X.shape[0] * self.subsample_input_size))
                samples_idx = np.random.choice(a=samples, size=size_x, replace=False)

                # set epoch time
                start_epoch = time.time()

                # optimize
                self.__optimize_comm(X=X[:, :self.cutting_point], y=y, P=P, E=E, A=A, B=B,
                                     samples_idx=samples_idx)
                self.__optimize_path(X=X, y=y, W=W, H=H, P=P, E=E, A=A, B=B, samples_idx=samples_idx,
                                     learning_rate=learning_rate)

                end_epoch = time.time()
                self.is_fit = True

                # Save models parameters based on test frequencies
                if (epoch % self.display_interval) == 0 or epoch == 1 or epoch == n_epochs - 1:
                    new_cost = self.__cost(M=None, X=X, y=y, P=P, E=E, A=A, B=B) / self.num_epochs
                    data = str(epoch) + '\t' + str(round(end_epoch - start_epoch, 3)) + '\t' + str(
                        new_cost) + '\t' + str(old_cost) + '\n'
                    save_data(data=data, file_name=cost_file_name, save_path=result_path, mode='a', w_string=True,
                              print_tag=False)
                    print('\t\t           --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_cost, old_cost))
                    logger.info('\t\t           --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_cost, old_cost))
                    if old_cost >= new_cost or epoch == n_epochs - 1:
                        C_name = model_name + "_C.pkl"
                        K_name = model_name + "_K.pkl"
                        T_name = model_name + "_T.pkl"
                        R_name = model_name + "_R.pkl"
                        L_name = model_name + "_L.pkl"
                        Z_name = model_name + "_Z.pkl"

                        if old_cost >= new_cost:
                            logger.info('\t\t  >> Storing C to: {0}'.format(C_name))
                            save_data(data=self.C, file_name=C_name, save_path=model_path, mode="wb", print_tag=False)
                            logger.info('\t\t  >> Storing K to: {0}'.format(K_name))
                            save_data(data=self.K, file_name=K_name, save_path=model_path, mode="wb", print_tag=False)
                            logger.info('\t\t  >> Storing T to: {0}'.format(T_name))
                            save_data(data=self.T, file_name=T_name, save_path=model_path, mode="wb", print_tag=False)
                            logger.info('\t\t  >> Storing R to: {0}'.format(R_name))
                            save_data(data=self.R, file_name=R_name, save_path=model_path, mode="wb", print_tag=False)
                            if not self.fit_pure_comm:
                                logger.info('\t\t  >> Storing L to: {0}'.format(L_name))
                                save_data(data=self.L, file_name=L_name, save_path=model_path, mode="wb", print_tag=False)
                            logger.info('\t\t  >> Storing Z to: {0}'.format(Z_name))
                            save_data(data=self.Z, file_name=Z_name, save_path=model_path, mode="wb", print_tag=False)
                        if epoch == n_epochs - 1:
                            C_name = model_name + "_C_final.pkl"
                            K_name = model_name + "_K_final.pkl"
                            T_name = model_name + "_T_final.pkl"
                            R_name = model_name + "_R_final.pkl"
                            L_name = model_name + "_L_final.pkl"
                            Z_name = model_name + "_Z_final.pkl"
                            logger.info('\t\t  >> Storing C to: {0}'.format(C_name))
                            save_data(data=self.C, file_name=C_name, save_path=model_path, mode="wb", print_tag=False)
                            logger.info('\t\t  >> Storing K to: {0}'.format(K_name))
                            save_data(data=self.K, file_name=K_name, save_path=model_path, mode="wb", print_tag=False)
                            logger.info('\t\t  >> Storing T to: {0}'.format(T_name))
                            save_data(data=self.T, file_name=T_name, save_path=model_path, mode="wb", print_tag=False)
                            logger.info('\t\t  >> Storing R to: {0}'.format(R_name))
                            save_data(data=self.R, file_name=R_name, save_path=model_path, mode="wb", print_tag=False)
                            if not self.fit_pure_comm:
                                logger.info('\t\t  >> Storing L to: {0}'.format(L_name))
                                save_data(data=self.L, file_name=L_name, save_path=model_path, mode="wb", print_tag=False)
                            logger.info('\t\t  >> Storing Z to: {0}'.format(Z_name))
                            save_data(data=self.Z, file_name=Z_name, save_path=model_path, mode="wb", print_tag=False)
                        del self.C, self.K, self.T, self.R, self.L, self.Z
                        if old_cost >= new_cost:
                            desc = '\t\t  >> Storing the triUMPF model to: {0:s}.pkl'.format(model_name)
                            print(desc)
                            logger.info(desc)
                            save_data(data=copy.copy(self), file_name=model_name + '.pkl', save_path=model_path,
                                      mode="wb", print_tag=False)
                        if epoch == n_epochs - 1:
                            desc = '\t\t  >> Storing the triUMPF model to: {0:s}.pkl'.format(model_name + '_final.pkl')
                            print(desc)
                            logger.info(desc)
                            save_data(data=copy.copy(self), file_name=model_name + '_final.pkl', save_path=model_path,
                                      mode="wb", print_tag=False)
                        if epoch != n_epochs - 1:
                            self.C = load_data(file_name=C_name, load_path=model_path, tag="triUMPF\'s C parameters",
                                               print_tag=False)
                            self.K = load_data(file_name=K_name, load_path=model_path, tag="triUMPF\'s K parameters",
                                               print_tag=False)
                            self.T = load_data(file_name=T_name, load_path=model_path, tag="triUMPF\'s T parameters",
                                               print_tag=False)
                            self.R = load_data(file_name=R_name, load_path=model_path, tag="triUMPF\'s R parameters",
                                               print_tag=False)
                            if not self.fit_pure_comm:
                                self.L = load_data(file_name=L_name, load_path=model_path, tag="triUMPF\'s L parameters",
                                                print_tag=False)
                            self.Z = load_data(file_name=Z_name, load_path=model_path, tag="triUMPF\'s Z parameters",
                                               print_tag=False)

                        if self.early_stop:
                            relative_change = np.abs((new_cost - old_cost) / old_cost)
                            desc = '\t\t    --> There is a little improvement in the cost '
                            desc += '(< {0}) for epoch {1}, hence, training is terminated...'.format(
                                self.loss_threshold, epoch)
                            if relative_change < self.loss_threshold:
                                logger.info(desc)
                                break
                        old_cost = new_cost
        print('\t  --> Training consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))
        logger.info('\t  --> Training consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))

    def inverse_transform(self, X1, X2=None, X3=None, comm=False):
        """Transform data back to its original space.
        Returns
        -------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix of original shape
            :param X3:
            :param display_params:
        """
        if not comm:
            assert X1.shape[1] == X2.shape[1]
        else:
            if X1 is None:
                raise Exception("Please provide a feature matrix.")
            assert X1.shape[0] == X2.shape[0]
            assert X1.shape[1] == X3.shape[0]
            assert X2.shape[1] == X3.shape[1]
            X2 = np.dot(X3, X2.T).T
        M_hat = np.dot(X1, X2.T)
        M_hat[M_hat < self.eps] = 0.
        return lil_matrix(M_hat)

    def __predict(self, X, batch_idx, total_progress):
        prob_label = np.zeros((X.shape[0], self.num_labels)) + EPSILON
        for label_idx in np.arange(self.num_labels):
            coef_intercept_label = self.coef_label[label_idx]
            if self.fit_intercept:
                coef_intercept_label = hstack((self.intercept_label[label_idx], coef_intercept_label))
            prob_label[:, label_idx] = self.__sigmoid(X.dot(coef_intercept_label.T).toarray().flatten())
        desc = '\t\t--> Computed {0:.4f}%...'.format(((batch_idx + 1) / total_progress * 100))
        print(desc, end="\r")
        return prob_label

    def predict(self, X, estimate_prob=False, apply_t_criterion=False, adaptive_beta=0.45,
                decision_threshold=0.5, top_k=-1, batch_size=30, num_jobs=1):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")

        desc = '\t>> Predicting multi-labels using triUMPF model...'
        print(desc)
        logger.info(desc)

        self.batch = batch_size
        self.decision_threshold = decision_threshold
        self.num_jobs = num_jobs

        if batch_size < 0:
            self.batch = 30
        if decision_threshold < 0:
            self.decision_threshold = 0.5
        if num_jobs < 0:
            self.num_jobs = 1

        # binarize or normalize features
        if self.binarize_input_feature:
            if self.use_external_features:
                X[:, :self.cutting_point] = preprocessing.binarize(X[:, :self.cutting_point])
            else:
                X = preprocessing.binarize(X)
        if self.normalize_input_feature:
            if self.use_external_features:
                X[:, :self.cutting_point] = preprocessing.normalize(X[:, :self.cutting_point])
            else:
                X = preprocessing.normalize(X)

        if self.fit_intercept:
            X = hstack((lil_matrix(np.ones((X.shape[0], 1))), X))

        X = lil_matrix(X)
        num_samples = X.shape[0]
        list_batches = np.arange(start=0, stop=num_samples, step=self.batch)
        parallel = Parallel(n_jobs=self.num_jobs, verbose=max(0, self.verbose - 1))
        prob_label = parallel(delayed(self.__predict)(X[batch:batch + self.batch],
                                                      batch_idx, len(list_batches))
                              for batch_idx, batch in enumerate(list_batches))
        desc = '\t\t--> Computed {0:.4f}%...'.format(100)
        logger.info(desc)
        print(desc)

        prob_label = np.vstack(prob_label)
        if apply_t_criterion and not estimate_prob:
            maxval = np.max(prob_label, axis=1) * adaptive_beta
            for sidx in np.arange(prob_label.shape[0]):
                prob_label[sidx][prob_label[sidx] >= maxval[sidx]] = 1

        if not estimate_prob:
            if top_k < 0:
                prob_label[prob_label >= self.decision_threshold] = 1
                prob_label[prob_label != 1] = 0
            else:
                prob_label[prob_label >= self.decision_threshold] = 1
                prob_label[prob_label != 1] = 0
        return lil_matrix(prob_label)

    def __top_groups(self, X, feature_names, top_k_features):
        X = X.T
        num_groups = X.shape[0]
        if feature_names is None:
            feature_names = np.arange(X.shape[1])
        groups_dict = dict()
        for group_idx in np.arange(num_groups):
            feats = [feature_names[i] for i in X[group_idx].argsort()[:-top_k_features - 1:-1]]
            groups_dict.update({group_idx: feats})
        return groups_dict

    def get_top_features(self, X, feature_names=None, top_k_features=5):
        clusters_dict = self.__top_groups(X, feature_names, top_k_features)
        return clusters_dict
