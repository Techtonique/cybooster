"""
examples.py
============

Self-contained worked examples for the closed-form Integrated Gradients
methodology, assuming `cybooster.SkBoosterRegressor` already exposes:

    .fit(X, y)                                   -> self
    .predict(X)                                  -> np.ndarray
    .get_sensitivities(X, columns=...)           -> DataFrame  (pointwise dF/dx_j)
    .get_feature_importances(X, columns=...)     -> DataFrame  (mean |sensitivity|)
    .get_summary(X, columns=...)                 -> DataFrame  (mean/std/CI/p-value)
    .get_integrated_gradients(X, X_baseline=None, columns=...) -> DataFrame  (closed-form IG)
    .plot_importance(X, columns=..., kind=...)               -> Figure
    .plot_beeswarm(X, columns=..., kind=...)                 -> Figure
    .plot_heterogeneity(X, columns=..., kind=..., smooth=..., frac=..., n_boot=..., ci=...) -> Figure

Two worked examples:
  1. Diabetes (sklearn built-in, no network dependency) -- the main case.
  2. Boston housing (classic benchmark, GitHub CSV mirror) -- kept for
     continuity with prior work, with an explicit caveat on its `b`
     variable (see the docstring of `run_boston`).

Each example: fits the model, runs a completeness check on the closed-form
IG (sum of attributions should equal F(x) - F(baseline) to numerical
precision), and produces three figures: global importance, beeswarm
(attribution + heterogeneity), and LOWESS-smoothed local heterogeneity for
the top features.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.datasets import load_diabetes

from cybooster import SkBoosterRegressor

OUTDIR = "."  # change if you want figures written elsewhere


# ----------------------------------------------------------------------------
# shared fit / diagnostics / figure-generation routine
# ----------------------------------------------------------------------------
def run_example(X, y, cols, name, top_k_features=3,
                 n_estimators=100, learning_rate=0.1, n_hidden_features=5,
                 activation="relu", reg_alpha=0.1, test_size=0.25, seed=123):
    """Fit, sanity-check, and plot one dataset. Returns the fitted model
    and the test split, in case further ad hoc inspection is wanted."""

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

    model = SkBoosterRegressor(
        obj=Ridge(alpha=reg_alpha, fit_intercept=False),
        n_estimators=n_estimators, learning_rate=learning_rate,
        n_hidden_features=n_hidden_features, activation=activation,
        col_sample=1.0, row_sample=1.0, direct_link=1, verbose=0, seed=seed,
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    print(f"\n=== {name} ===")
    print(f"Test R2:  {r2_score(yte, pred):.4f}")
    print(f"Test MAE: {mean_absolute_error(yte, pred):.4f}")

    # --- closed-form Integrated Gradients + completeness check ---
    ig = model.get_integrated_gradients(Xte, columns=cols)
    baseline = np.tile(model.fit_obj_["xm"], (Xte.shape[0], 1))
    baseline_pred = model.predict(baseline)
    completeness_err = np.max(np.abs(ig.sum(axis=1).values - (pred - baseline_pred)))
    print(f"IG completeness max abs error: {completeness_err:.2e}  (should be ~machine eps)")

    # --- global importance (mean |IG|) ---
    print("\nMean |IG| per feature:")
    print(ig.abs().mean().sort_values(ascending=False).round(3))

    # --- figures ---
    fig_imp = model.plot_importance(Xte, columns=cols, kind="ig")
    fig_imp.savefig(f"{OUTDIR}/{name}_importance.png", bbox_inches="tight")

    fig_bee = model.plot_beeswarm(Xte, columns=cols, kind="ig")
    fig_bee.savefig(f"{OUTDIR}/{name}_beeswarm.png", bbox_inches="tight")

    fig_het = model.plot_heterogeneity(Xte, columns=cols, kind="ig",
                                        top_k=top_k_features, smooth=True,
                                        frac=0.4, n_boot=200, ci=0.90)
    fig_het.savefig(f"{OUTDIR}/{name}_heterogeneity.png", bbox_inches="tight")

    print(f"Saved: {name}_importance.png, {name}_beeswarm.png, {name}_heterogeneity.png")
    return model, (Xtr, Xte, ytr, yte)


# ----------------------------------------------------------------------------
# example 1: diabetes (main case -- no network dependency, no ethical baggage)
# ----------------------------------------------------------------------------
def run_diabetes():
    data = load_diabetes(as_frame=True)
    X = data.data.values.astype(np.float64)
    y = data.target.values.astype(np.float64)
    cols = data.data.columns.tolist()
    return run_example(X, y, cols, name="diabetes", top_k_features=3)


# ----------------------------------------------------------------------------
# example 2: Boston housing (continuity case)
#
# Kept here for comparability with earlier results in this line of work, not
# as an endorsement of the dataset. The `b` column is 1000*(Bk-0.63)^2, a
# variable constructed around an assumption that racial composition affects
# property values non-monotonically; scikit-learn removed `load_boston` over
# this and related issues (see Carlisle, "Racist data destruction?", 2019).
# Results below should be read purely as a demonstration of the attribution
# method, not as a claim about the housing market.
# ----------------------------------------------------------------------------
def run_boston():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    df["chas"] = df["chas"].astype(int)
    y = df["medv"].values.astype(np.float64)
    X = df.drop(columns=["medv"]).values.astype(np.float64)
    cols = df.drop(columns=["medv"]).columns.tolist()
    return run_example(X, y, cols, name="boston", top_k_features=3)


if __name__ == "__main__":
    run_diabetes()
    run_boston()