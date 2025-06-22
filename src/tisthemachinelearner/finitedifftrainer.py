from .base import BaseModel
from sklearn.base import RegressorMixin
from copy import deepcopy
from collections import namedtuple
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import norm
import nnetsauce as ns  # adjust if your import path differs
import pandas as pd
# import your matrix operations helper if needed (mo.rbind)

class FiniteDiffRegressor(BaseModel, RegressorMixin):
    def __init__(self, base_model, lr=1e-4, optimizer='gd', eps=1e-3, batch_size=32, **kwargs):
        """
        Finite difference trainer for nnetsauce models.

        Args:
            model: trained nnetsauce model.
            lr: learning rate.
            optimizer: 'gd' (gradient descent) 'sgd' (stochastic gradient descent) or 'adam' or 'cd' (coordinate descent).
            eps: scaling factor for adaptive finite difference step size.
            **kwargs: Additional parameters to pass to the scikit-learn model.
        """
        super().__init__(base_model, True, **kwargs)
        self.model = ns.CustomRegressor(self.model, **self.custom_kwargs)
        assert isinstance(self.model, ns.CustomRegressor),\
         "'model' must be of class ns.CustomRegressor"
        self.lr = lr
        self.optimizer = optimizer
        self.eps = eps
        self.loss_history_ = []
        self.opt_state = None
        self.batch_size = batch_size  # for SGD
        self.loss_history_ = []
        self._cd_index = 0  # For coordinate descent

    def _loss(self, X, y):
        y_pred = self.model.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))

    def _compute_grad(self, X, y):
        W = deepcopy(self.model.W_)
        shape = W.shape
        W_flat = W.flatten()
        n_params = W_flat.size

        # Adaptive finite difference step
        h_vec = self.eps * np.maximum(1.0, np.abs(W_flat))
        eye = np.eye(n_params)

        loss_plus = np.zeros(n_params)
        loss_minus = np.zeros(n_params)

        for i in range(n_params):
            h_i = h_vec[i]
            Wp = W_flat.copy(); Wp[i] += h_i
            Wm = W_flat.copy(); Wm[i] -= h_i

            self.model.W_ = Wp.reshape(shape)
            loss_plus[i] = self._loss(X, y)

            self.model.W_ = Wm.reshape(shape)
            loss_minus[i] = self._loss(X, y)

        grad = ((loss_plus - loss_minus) / (2 * h_vec)).reshape(shape)
        self.model.W_ = W  # restore original
        return grad

    def fit(self, X, y, sample_weight=None, epochs=10, verbose=True, show_progress=True, **kwargs):
        """
        Optimizes W_ using finite differences and retrains readout.

        Args:
            X, y: data to compute loss and retrain output
            epochs: number of optimization steps
            verbose: whether to print progress messages
            show_progress: whether to show tqdm progress bar

        Returns:
            self (enables method chaining)
        """
        # Initial fit of output layer with current W_
        # Final retrain after updates
        self.model.fit(X, y)

        iterator = tqdm(range(epochs)) if show_progress else range(epochs)

        for epoch in iterator:
            grad = self._compute_grad(X, y)

            if self.optimizer == 'gd':
                self.model.W_ -= self.lr * grad
                self.model.W_ = np.clip(self.model.W_, 0, 1)
                #print("self.model.W_", self.model.W_)

            elif self.optimizer == 'sgd':
                # Sample a mini-batch for stochastic gradient
                n_samples = X.shape[0]
                idxs = np.random.choice(n_samples, self.batch_size, replace=False)
                if isinstance(X, pd.DataFrame):
                    X_batch = X.iloc[idxs,:]
                    y_batch = y[idxs]
                else: 
                    X_batch = X[idxs,:]
                    y_batch = y[idxs]
                grad = self._compute_grad(X_batch, y_batch)

                self.model.W_ -= self.lr * grad
                self.model.W_ = np.clip(self.model.W_, 0, 1)

            elif self.optimizer == 'adam':
                if self.opt_state is None:
                    self.opt_state = {'m': np.zeros_like(grad), 'v': np.zeros_like(grad), 't': 0}
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                self.opt_state['t'] += 1
                self.opt_state['m'] = beta1 * self.opt_state['m'] + (1 - beta1) * grad
                self.opt_state['v'] = beta2 * self.opt_state['v'] + (1 - beta2) * (grad ** 2)
                m_hat = self.opt_state['m'] / (1 - beta1 ** self.opt_state['t'])
                v_hat = self.opt_state['v'] / (1 - beta2 ** self.opt_state['t'])

                self.model.W_ -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
                self.model.W_ = np.clip(self.model.W_, 0, 1)
                #print("self.model.W_", self.model.W_)

            elif self.optimizer == 'cd':  # coordinate descent

                W_shape = self.model.W_.shape
                W_flat_size = self.model.W_.size
                W_flat = self.model.W_.flatten()
                grad_flat = grad.flatten()

                # Update only one coordinate per epoch (cyclic)
                idx = self._cd_index % W_flat_size
                W_flat[idx] -= self.lr * grad_flat[idx]
                # Clip the updated value
                W_flat[idx] = np.clip(W_flat[idx], 0, 1)

                # Restore W_
                self.model.W_ = W_flat.reshape(W_shape)

                self._cd_index += 1

            else:
                raise ValueError(f"Unsupported optimizer: {self.optimizer}")

            loss = self._loss(X, y)
            self.loss_history_.append(loss)

            if verbose:
                print(f"Epoch {epoch+1}: Loss = {loss:.6f}")                

        # if sample_weights, else: (must use self.row_index)
        if sample_weight is not None:
            self.model.fit(
                X,
                y,
                sample_weight=sample_weight[self.index_row_].ravel(),
                **kwargs
            )

            return self

        return self


    def predict(self, X, level=95, method='splitconformal', **kwargs):
        """Predict test data X.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            level: int
                Level of confidence (default = 95)

            method: str
                'splitconformal', 'localconformal'
                prediction (if you specify `return_pi = True`)

            **kwargs: additional parameters
                    `return_pi = True` for conformal prediction,
                    with `method` in ('splitconformal', 'localconformal')
                    or `return_std = True` for `self.model` in
                    (`sklearn.linear_model.BayesianRidge`,
                    `sklearn.linear_model.ARDRegressor`,
                    `sklearn.gaussian_process.GaussianProcessRegressor`)`

        Returns:

            model predictions:
                an array if uncertainty quantification is not requested,
                  or a tuple if with prediction intervals and simulations
                  if `return_std = True` (mean, standard deviation,
                  lower and upper prediction interval) or `return_pi = True`
                  ()

        """

        if "return_std" in kwargs:

            alpha = 100 - level
            pi_multiplier = norm.ppf(1 - alpha / 200)

            if len(X.shape) == 1:

                n_features = X.shape[0]
                new_X = mo.rbind(
                    X.reshape(1, n_features),
                    np.ones(n_features).reshape(1, n_features),
                )

                mean_, std_ = self.model.predict(
                    new_X, return_std=True
                )[0]

                preds =  mean_
                lower =  (mean_ - pi_multiplier * std_)
                upper =  (mean_ + pi_multiplier * std_)

                DescribeResults = namedtuple(
                    "DescribeResults", ["mean", "std", "lower", "upper"]
                )

                return DescribeResults(preds, std_, lower, upper)

            # len(X.shape) > 1
            mean_, std_ = self.model.predict(
                X, return_std=True
            )

            preds =  mean_
            lower =  (mean_ - pi_multiplier * std_)
            upper =  (mean_ + pi_multiplier * std_)

            DescribeResults = namedtuple(
                "DescribeResults", ["mean", "std", "lower", "upper"]
            )

            return DescribeResults(preds, std_, lower, upper)

        if "return_pi" in kwargs:
            assert method in (
                "splitconformal",
                "localconformal",
            ), "method must be in ('splitconformal', 'localconformal')"
            self.pi = ns.PredictionInterval(
                obj=self,
                method=method,
                level=level,
                type_pi=self.type_pi,
                replications=self.replications,
                kernel=self.kernel,
            )

            if len(self.X_.shape) == 1:
                if isinstance(X, pd.DataFrame):
                    self.X_ = pd.DataFrame(
                        self.X_.values.reshape(1, -1), columns=self.X_.columns
                    )
                else:
                    self.X_ = self.X_.reshape(1, -1)
                self.y_ = np.array([self.y_])

            self.pi.fit(self.X_, self.y_)
            # self.X_ = None # consumes memory to keep, dangerous to delete (side effect)
            # self.y_ = None # consumes memory to keep, dangerous to delete (side effect)
            preds = self.pi.predict(X, return_pi=True)
            return preds

        # "return_std" not in kwargs
        if len(X.shape) == 1:

            n_features = X.shape[0]
            new_X = mo.rbind(
                X.reshape(1, n_features),
                np.ones(n_features).reshape(1, n_features),
            )

            return (
                0
                + self.model.predict(new_X, **kwargs)
            )[0]

        # len(X.shape) > 1
        return  self.model.predict(
            X, **kwargs
        )