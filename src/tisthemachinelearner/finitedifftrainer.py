import nnetsauce as ns  # adjust if your import path differs
import pandas as pd
import numpy as np

from .base import BaseModel
from sklearn.base import RegressorMixin
from copy import deepcopy
from collections import namedtuple
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import norm
from sklearn.metrics import mean_pinball_loss
# import your matrix operations helper if needed (mo.rbind)

class FiniteDiffRegressor(BaseModel, RegressorMixin):
    """
    Finite difference trainer for nnetsauce models.

    Parameters
    ----------

    base_model : str
        The name of the base model (e.g., 'RidgeCV').

    lr : float, optional
        Learning rate for optimization (default=1e-4).

    optimizer : {'gd', 'sgd', 'adam', 'cd'}, optional
        Optimization algorithm: gradient descent ('gd'), stochastic gradient descent ('sgd'),
        Adam ('adam'), or coordinate descent ('cd'). Default is 'gd'.

    eps : float, optional
        Scaling factor for adaptive finite difference step size (default=1e-3).

    batch_size : int, optional
        Batch size for 'sgd' optimizer (default=32).

    alpha : float, optional
        Elastic net penalty strength (default=0.0).

    l1_ratio : float, optional
        Elastic net mixing parameter (0 = Ridge, 1 = Lasso, default=0.0).

    type_loss : {'mse', 'quantile'}, optional
        Type of loss function to use (default='mse').

    q : float, optional
        Quantile for quantile loss (default=0.5).

    **kwargs
        Additional parameters to pass to the scikit-learn model.

    """

    def __init__(self, base_model, 
                lr=1e-4, optimizer='gd', 
                eps=1e-4, batch_size=32, 
                alpha=0.0, l1_ratio=0.0, 
                type_loss="mse", q=0.5,
                **kwargs):
        super().__init__(base_model, True, **kwargs)
        self.model = ns.CustomRegressor(self.model, **self.custom_kwargs)
        assert isinstance(self.model, ns.CustomRegressor), \
            "'model' must be of class ns.CustomRegressor"
        
        # Hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.eps = eps
        self.batch_size = batch_size
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.type_loss = type_loss
        self.q = q
        
        # Training state
        self.loss_history_ = []
        self.opt_state = None
        self._cd_index = 0
        self._is_initialized = False

    def _initialize_weights(self, X):
        """Initialize weights using proper neural network initialization"""
        input_dim = X.shape[1]
        
        # Get model architecture details
        n_hidden = getattr(self.model, 'n_hidden_features', 1)
        n_clusters = getattr(self.model, 'n_clusters', -1)
        
        # Determine weight shape
        if n_clusters >= 0:
            shape = (input_dim, n_hidden + n_clusters)
        else:
            shape = (input_dim, n_hidden)
        
        # He initialization (good for ReLU-like activations)
        scale = np.sqrt(2.0 / input_dim)
        self.model.W_ = np.random.normal(0, scale, size=shape)
        self._is_initialized = True

    def _loss(self, X, y, **kwargs):
        """Compute loss with elastic net penalty"""
        y_pred = self.model.predict(X)
        
        if self.type_loss == "mse":
            loss = np.mean((y - y_pred) ** 2)
        elif self.type_loss == "quantile":
            loss = mean_pinball_loss(y, y_pred, alpha=self.q, **kwargs)
        
        # Elastic net regularization
        W = self.model.W_
        l1 = np.sum(np.abs(W))
        l2 = np.sum(W ** 2)
        reg = self.alpha * (self.l1_ratio * l1 + 0.5 * (1 - self.l1_ratio) * l2)
        
        return loss + reg

    def _compute_grad(self, X, y):
        """Compute gradient using finite differences"""
        if not self._is_initialized:
            self._initialize_weights(X)
            
        W = self.model.W_.copy()  # Use current weights
        shape = W.shape
        W_flat = W.flatten()
        n_params = W_flat.size

        # Adaptive finite difference step
        h_vec = self.eps * np.maximum(1.0, np.abs(W_flat))
        
        # Central difference calculation
        loss_plus = np.zeros(n_params)
        loss_minus = np.zeros(n_params)

        for i in range(n_params):
            h_i = h_vec[i]
            
            # Positive perturbation
            W_plus = W_flat.copy()
            W_plus[i] += h_i
            self.model.W_ = W_plus.reshape(shape)
            loss_plus[i] = self._loss(X, y)

            # Negative perturbation
            W_minus = W_flat.copy()
            W_minus[i] -= h_i
            self.model.W_ = W_minus.reshape(shape)
            loss_minus[i] = self._loss(X, y)

        # Central difference gradient with numerical stability
        grad = ((loss_plus - loss_minus) / (2 * h_vec + 1e-12)).reshape(shape)
        
        # Add elastic net gradient components
        grad += self.alpha * self.l1_ratio * np.sign(W)
        grad += self.alpha * (1 - self.l1_ratio) * W
        
        # Restore original weights
        self.model.W_ = W
        return grad

    def fit(self, X, y, epochs=10, verbose=True, show_progress=True, sample_weight=None, **kwargs):
        """Fit model using finite difference optimization"""
        # Initialize weights if not already done
        if not self._is_initialized:
            self._initialize_weights(X)
        
        # Training loop
        iterator = tqdm(range(epochs)) if show_progress else range(epochs)
        
        for epoch in iterator:
            # Compute gradient and update weights
            grad = self._compute_grad(X, y)
            
            if self.optimizer == 'gd':
                self.model.W_ -= self.lr * grad
                
            elif self.optimizer == 'sgd':
                # Mini-batch gradient
                idxs = np.random.choice(X.shape[0], self.batch_size, replace=False)
                X_batch = X.iloc[idxs] if isinstance(X, pd.DataFrame) else X[idxs]
                y_batch = y[idxs]
                grad = self._compute_grad(X_batch, y_batch)
                self.model.W_ -= self.lr * grad
                
            elif self.optimizer == 'adam':
                # Initialize Adam state if needed
                if self.opt_state is None:
                    self.opt_state = {
                        'm': np.zeros_like(grad),
                        'v': np.zeros_like(grad),
                        't': 0
                    }
                
                # Adam update
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                self.opt_state['t'] += 1
                self.opt_state['m'] = beta1 * self.opt_state['m'] + (1 - beta1) * grad
                self.opt_state['v'] = beta2 * self.opt_state['v'] + (1 - beta2) * (grad ** 2)
                
                # Bias correction
                m_hat = self.opt_state['m'] / (1 - beta1 ** self.opt_state['t'])
                v_hat = self.opt_state['v'] / (1 - beta2 ** self.opt_state['t'])
                
                self.model.W_ -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
                
            elif self.optimizer == 'cd':
                # Coordinate descent
                W_flat = self.model.W_.flatten()
                grad_flat = grad.flatten()
                idx = self._cd_index % len(W_flat)
                W_flat[idx] -= self.lr * grad_flat[idx]
                self.model.W_ = W_flat.reshape(self.model.W_.shape)
                self._cd_index += 1
                
            else:
                raise ValueError(f"Unsupported optimizer: {self.optimizer}")

            # Track loss
            loss = self._loss(X, y)
            self.loss_history_.append(loss)
            
            if verbose and (epoch % max(1, epochs//10) == 0):
                print(f"Epoch {epoch+1}/{epochs}: Loss = {loss:.6f}")

        # Handle sample weights if provided
        if sample_weight is not None:
            self.model.fit(
                X,
                y,
                sample_weight=sample_weight[self.index_row_].ravel(),
                **kwargs
            )

        return self

    def predict(self, X, level=95, method='splitconformal', **kwargs):
        """
        Predict using the trained model.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Input data.

        level : int, optional
            Level of confidence for prediction intervals (default=95).

        method : {'splitconformal', 'localconformal'}, optional
            Method for conformal prediction (default='splitconformal').

        **kwargs
            Additional keyword arguments. Use `return_pi=True` for prediction intervals,
            or `return_std=True` for standard deviation estimates.

        Returns
        -------
        
        array or tuple
            Model predictions, or a tuple with prediction intervals or standard deviations if requested.
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
