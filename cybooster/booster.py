import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from ._boosterc import (
    BoosterRegressor, BoosterClassifier,
    fit_booster_regressor, predict_booster_regressor,
)

# ----------------------------------------------------------------------------
# activation functions + derivatives, shared by sensitivities and IG
# ----------------------------------------------------------------------------
_ACTIVATIONS = {
    "relu":    (lambda z: np.maximum(z, 0),          lambda z: (z > 0).astype(float)),
    "relu6":   (lambda z: np.clip(z, 0, 6),           lambda z: ((z > 0) & (z < 6)).astype(float)),
    "tanh":    (np.tanh,                              lambda z: 1 - np.tanh(z) ** 2),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)),        lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))),
}


def _secant_slope(g, z0, z1, g_prime, eps=1e-9):
    """Average derivative of g over [z0, z1] via FTC; falls back to g'(z0) when z0≈z1."""
    dz = z1 - z0
    flat = np.abs(dz) < eps
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (g(z1) - g(z0)) / np.where(flat, 1.0, dz)
    return np.where(flat, g_prime(z0), raw)


# ----------------------------------------------------------------------------
# plot styling, shared by all plotting methods below
# ----------------------------------------------------------------------------
_PALETTE = {"bg": "#fbfaf7", "ink": "#1f2430", "grid": "#e4e1d8",
            "pos": "#c1440e", "neg": "#2f6f6f", "muted": "#9a9488"}

plt.rcParams.update({
    "figure.facecolor": _PALETTE["bg"], "axes.facecolor": _PALETTE["bg"],
    "axes.edgecolor": _PALETTE["muted"], "axes.labelcolor": _PALETTE["ink"],
    "axes.grid": True, "grid.color": _PALETTE["grid"], "grid.linewidth": 0.8,
    "text.color": _PALETTE["ink"], "xtick.color": _PALETTE["ink"], "ytick.color": _PALETTE["ink"],
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": _PALETTE["bg"], "savefig.dpi": 160,
})


def _diverging_cmap(z):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("teal_orange", [_PALETTE["neg"], "#e8e3d3", _PALETTE["pos"]])
    return cmap(np.clip((z + 2.5) / 5.0, 0, 1))


class SkBoosterRegressor(BoosterRegressor, BaseEstimator, RegressorMixin):
    """A scikit-learn compatible wrapper for BoosterRegressor.

    Adds, on top of the compiled class's own .fit/.predict/.get_sensitivities/
    .get_feature_importances/.get_summary/.update:

      - a Python-level __init__ that stores hyperparameters as public
        attributes, which BaseEstimator.get_params()/set_params()/clone()
        require (the compiled __init__'s parameters aren't introspectable
        by sklearn otherwise -- get_params() raises RuntimeError as shipped).
      - .get_integrated_gradients(): closed-form, baseline-relative
        attribution (exact -- no numerical quadrature), which the compiled
        class can't offer since it doesn't expose its internal per-round
        state (col_index_i / W_i / fit_obj_i) to Python.
      - .plot_importance() / .plot_beeswarm() / .plot_heterogeneity():
        built on .get_sensitivities() by default (kind="gradient"), since
        that needs no baseline choice; pass kind="ig" for baseline-relative
        attribution plots instead.

    Note: .fit() currently calls the underlying boosting routine twice --
    once via the compiled BoosterRegressor.fit() (so update()/predict()/
    get_sensitivities()/get_summary() keep working exactly as before), and
    once more here to populate self.fit_obj_ as a plain Python dict (needed
    for get_integrated_gradients()). Both calls use the same `seed`, so
    they're numerically identical, but this doubles fit cost. The proper
    fix is upstream: expose the compiled class's internal `fit_obj` dict
    (e.g. as `cdef public dict fit_obj`) so this wrapper can reuse it
    directly instead of re-fitting.
    """

    def __init__(self, obj, n_estimators=100, learning_rate=0.1, n_hidden_features=5,
                 reg_lambda=0.1, alpha=0.5, row_sample=1.0, col_sample=1.0, dropout=0.0,
                 tolerance=1e-6, direct_link=1, verbose=1, seed=123, backend="cpu",
                 activation="relu", weights_distr="uniform"):
        super().__init__(obj, n_estimators, learning_rate, n_hidden_features, reg_lambda,
                          alpha, row_sample, col_sample, dropout, tolerance, direct_link,
                          verbose, seed, backend, activation, weights_distr)
        # public, sklearn-visible copies (names must match __init__ args exactly)
        self.obj = obj
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_hidden_features = n_hidden_features
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.row_sample = row_sample
        self.col_sample = col_sample
        self.dropout = dropout
        self.tolerance = tolerance
        self.direct_link = direct_link
        self.verbose = verbose
        self.seed = seed
        self.backend = backend
        self.activation = activation
        self.weights_distr = weights_distr
        self.fit_obj_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64")
        super().fit(X, y)  # populates the compiled class's private fit_obj

        # re-fit via the module-level function to get a Python-visible dict
        # (see docstring: same seed => numerically identical result)
        self.fit_obj_ = fit_booster_regressor(
            X=X, y=y, n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            n_hidden_features=self.n_hidden_features, reg_lambda=self.reg_lambda,
            alpha=self.alpha, row_sample=self.row_sample, col_sample=self.col_sample,
            dropout=self.dropout, tolerance=self.tolerance, direct_link=self.direct_link,
            verbose=0, seed=self.seed, backend=self.backend,
            activation=self.activation, weights_distr=self.weights_distr, obj=self.obj,
        )
        return self

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype="float64")
        return super().predict(X)

    def _check_fitted(self):
        if self.fit_obj_ is None:
            raise ValueError("Model not fitted yet. Call .fit(X, y) first.")

    # ------------------------------------------------------ integrated grad
    def get_integrated_gradients(self, X, X_baseline=None, columns=None):
        """
        Closed-form Integrated Gradients (exact, no numerical quadrature).
        Baseline-relative: sum_j IG_j == predict(X) - predict(X_baseline).
        Default baseline is the training feature means.
        """
        self._check_fitted()
        res = self.fit_obj_
        X = np.ascontiguousarray(X, dtype="float64")
        n, p = X.shape
        xm, sigma = res["xm"], res["xsd"]
        if X_baseline is None:
            X_baseline = np.tile(xm, (n, 1))
        X_std, X0_std = (X - xm) / sigma, (X_baseline - xm) / sigma
        g, g_prime = _ACTIVATIONS[res["activation"]]
        ig = np.zeros((n, p))

        for m in range(res["n_estimators"]):
            iy = res["col_index_i"][m]
            W_i = res["W_i"][m]
            coef = np.asarray(res["fit_obj_i"][m].coef_).ravel()
            n_direct = len(iy) if self.direct_link else 0
            z0 = X0_std[:, iy] @ W_i
            z1 = X_std[:, iy] @ W_i
            slope = _secant_slope(g, z0, z1, g_prime)
            for j_idx, j in enumerate(iy):
                direct = coef[j_idx] if self.direct_link else 0.0
                hidden = (coef[n_direct:] * slope * W_i[j_idx, :]).sum(axis=1)
                ig[:, j] += res["learning_rate"] * (direct + hidden) / max(sigma[j], 1e-6)

        ig = ig * (X - X_baseline)
        cols = columns if columns is not None else [f"x{j}" for j in range(p)]
        return pd.DataFrame(ig, columns=cols)

    # -------------------------------------------------------------- plots
    def _attr(self, X, columns, kind):
        if kind == "gradient":
            return self.get_sensitivities(X, columns=columns, show_progress=False)
        elif kind == "ig":
            return self.get_integrated_gradients(X, columns=columns)
        raise ValueError("kind must be 'gradient' or 'ig'")

    def plot_importance(self, X, columns=None, kind="gradient", ax=None):
        attr = self._attr(X, columns, kind)
        imp = attr.abs().mean().sort_values(ascending=True)
        label = "mean |sensitivity|  (dF/dx)" if kind == "gradient" else "mean |Integrated Gradient|"

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(imp) + 1.5))
        colors = [_PALETTE["pos"] if v == imp.max() else _PALETTE["neg"] for v in imp.values]
        bars = ax.barh(imp.index, imp.values, color=colors, height=0.62, zorder=3)
        for b, v in zip(bars, imp.values):
            ax.text(v + imp.max() * 0.015, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                    va="center", fontsize=9.5)
        ax.set_xlabel(label)
        ax.set_title("Global feature importance", pad=12)
        ax.grid(axis="y", visible=False)
        if fig is not None:
            fig.tight_layout()
        return fig or ax.figure

    def plot_beeswarm(self, X, columns=None, kind="gradient", jitter=0.3, seed=0, ax=None):
        attr = self._attr(X, columns, kind)
        X_df = pd.DataFrame(np.ascontiguousarray(X), columns=attr.columns)
        label = "sensitivity  dF/dx  (local marginal effect)" if kind == "gradient" \
            else "Integrated Gradient contribution"

        rng = np.random.default_rng(seed)
        order = attr.abs().mean().sort_values(ascending=True).index.tolist()
        Xz = (X_df - X_df.mean()) / X_df.std()

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(order) + 1.5))
        for i, feat in enumerate(order):
            y_jit = i + rng.uniform(-jitter, jitter, size=len(attr))
            colors = _diverging_cmap(Xz[feat].values)
            ax.scatter(attr[feat], y_jit, c=colors, s=16, alpha=0.85, linewidths=0, zorder=3)
        ax.axvline(0, color=_PALETTE["ink"], lw=1.0, alpha=0.5, zorder=2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlabel(label)
        ax.set_title("Attribution + heterogeneity, per observation", pad=12)
        ax.grid(axis="y", visible=False)

        sm = plt.cm.ScalarMappable(
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                "teal_orange", [_PALETTE["neg"], "#e8e3d3", _PALETTE["pos"]]),
            norm=matplotlib.colors.Normalize(vmin=-2.5, vmax=2.5))
        cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, ticks=[-2, 0, 2])
        cbar.ax.set_yticklabels(["low", "avg", "high"])
        cbar.set_label("feature value", fontsize=10)
        if fig is not None:
            fig.tight_layout()
        return fig or ax.figure

    def plot_heterogeneity(self, X, features=None, top_k=3, columns=None, kind="gradient",
                            smooth=True, frac=0.4, n_boot=200, ci=0.90, seed=0):
        """
        Scatter of local effect (sensitivity or IG) vs. each feature's own
        value, with an optional LOWESS trend line summarizing the shape of
        the heterogeneity (not just showing it as unstructured scatter).

        smooth : bool
            Overlay a LOWESS curve (statsmodels) through each panel.
        frac : float
            LOWESS span -- fraction of points used in each local fit.
            Lower = wigglier/more local, higher = smoother/more global.
        n_boot : int
            Bootstrap resamples used to draw a confidence band around the
            LOWESS curve. Set to 0 to skip the band (just the line).
        ci : float
            Confidence level for the band (e.g. 0.90 -> 5th/95th percentile).
        """
        attr = self._attr(X, columns, kind)
        if features is None:
            features = attr.abs().mean().sort_values(ascending=False).index[:top_k].tolist()
        X_df = pd.DataFrame(np.ascontiguousarray(X), columns=attr.columns)
        ylabel = "sensitivity  dF/dx" if kind == "gradient" else "IG contribution"
        rng = np.random.default_rng(seed)

        fig, axes = plt.subplots(1, len(features), figsize=(4.6 * len(features), 4.3))
        if len(features) == 1:
            axes = [axes]
        for ax, feat in zip(axes, features):
            x = X_df[feat].values
            yv = attr[feat].values
            ax.scatter(x, yv, s=20, alpha=0.55, color=_PALETTE["neg"],
                       edgecolors="none", zorder=3)
            ax.axhline(0, color=_PALETTE["muted"], lw=0.9, zorder=2)

            if smooth and len(x) > 5:
                grid = np.linspace(x.min(), x.max(), 200)

                if n_boot > 0:
                    boot_curves = np.empty((n_boot, len(grid)))
                    n = len(x)
                    for b in range(n_boot):
                        idx = rng.integers(0, n, n)
                        fit = lowess(yv[idx], x[idx], frac=frac, xvals=grid)
                        boot_curves[b] = fit
                    lo = np.nanpercentile(boot_curves, 100 * (1 - ci) / 2, axis=0)
                    hi = np.nanpercentile(boot_curves, 100 * (1 + ci) / 2, axis=0)
                    mid = np.nanmedian(boot_curves, axis=0)
                    ax.fill_between(grid, lo, hi, color=_PALETTE["pos"], alpha=0.15, zorder=1)
                    ax.plot(grid, mid, color=_PALETTE["pos"], lw=2.2, zorder=4)
                else:
                    fit = lowess(yv, x, frac=frac, xvals=grid)
                    ax.plot(grid, fit, color=_PALETTE["pos"], lw=2.2, zorder=4)

            ax.set_xlabel(feat)
            ax.set_ylabel(ylabel if feat == features[0] else "")
            ax.set_title(f"{feat}", pad=10)
        title = "Local effect heterogeneity" + (f"  (LOWESS trend, {int(ci*100)}% band)" if smooth and n_boot > 0 else "")
        fig.suptitle(title, y=1.03, fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


class SkBoosterClassifier(BoosterClassifier, BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for BoosterClassifier."""

    def __init__(self, obj, n_estimators=100, learning_rate=0.1, n_hidden_features=5,
                 reg_lambda=0.1, alpha=0.5, row_sample=1.0, col_sample=1.0, dropout=0.0,
                 tolerance=1e-6, direct_link=1, verbose=1, seed=123, backend="cpu",
                 activation="relu", weights_distr="uniform"):
        super().__init__(obj, n_estimators, learning_rate, n_hidden_features, reg_lambda,
                          alpha, row_sample, col_sample, dropout, tolerance, direct_link,
                          verbose, seed, backend, activation, weights_distr)
        self.obj = obj
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_hidden_features = n_hidden_features
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.row_sample = row_sample
        self.col_sample = col_sample
        self.dropout = dropout
        self.tolerance = tolerance
        self.direct_link = direct_link
        self.verbose = verbose
        self.seed = seed
        self.backend = backend
        self.activation = activation
        self.weights_distr = weights_distr

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype="float64")
        y = np.asarray(y, dtype="int64")
        super().fit(X, y)
        return self

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype="float64")
        return super().predict(X)

    def predict_proba(self, X):
        X = np.ascontiguousarray(X, dtype="float64")
        return super().predict_proba(X)  # was a bare `return` -- fixed