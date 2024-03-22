import tensorflow as tf
import tensorflow_probability as tfp
import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tfd = tfp.distributions
Root = tfd.JointDistributionCoroutine.Root


def gen_fourier_basis(n_t, p, n=3):
    """
    fourier basis of frequency [1,n]/p with sin & cosine
    """
    t = np.arange(n_t)
    x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
    return np.concatenate((np.cos(x), np.sin(x)), axis=1)


def gam_trend_seasonality(
    X_changepoint, X_seasonality, s_changepoint_times, n_t, ts_mean
):
    n_basis_funcs = X_seasonality.shape[1]
    beta = yield Root(
        tfd.Sample(tfd.Normal(0, .2), sample_shape=n_basis_funcs, name="beta")
    )  # (Nb)
    y_season = tf.einsum("ij,...j->...i", X_seasonality, beta)

    k = yield Root(tfd.HalfNormal(10.0, name="k"))
    m = yield Root(tfd.Normal(ts_mean, scale=5.0, name="m"))
    tau = yield Root(tfd.HalfNormal(10.0, name="tau"))
    delta = yield tfd.Sample(
        tfd.Laplace(0.0, tau), sample_shape=s_changepoint_times.shape, name="delta"
    )
    gamma = -s_changepoint_times * delta

    growth_rate = k[..., None] + tf.einsum("ij,...j->...i", X_changepoint, delta)
    y_offset = m[..., None] + tf.einsum("ij,...j->...i", X_changepoint, gamma)

    t = tf.cast(tf.linspace(0, 1, n_t), np.float32)
    y_trend = growth_rate * t + y_offset

    # Filter to only training data
    noise_sigma = yield Root(tfd.HalfNormal(scale=5.0, name="noise_sigma"))
    return y_season, y_trend, noise_sigma


def gam_input(num_steps_all, num_forecast_steps, n_changepoints=12, n_basis_funcs=6):
    # Build trend design matrix (X_cp)
    t = np.linspace(0, 1, num_steps_all, dtype=np.float32)
    s = np.linspace(0, max(t), n_changepoints + 2, dtype=np.float32)[1:-1]
    X_cp_all = (t[:, None] > s).astype(np.float32)
    X_cp = X_cp_all[:-num_forecast_steps, :]

    # Build seasonality is season-indicator matrix (X_sea)
    X_season_all = gen_fourier_basis(num_steps_all, p=12, n=n_basis_funcs)
    X_season = X_season_all[:-num_forecast_steps, :]  # (tp, Nb)
    n_basis_funcs = X_season_all.shape[-1]

    ## fig,ax=plt.subplots(1,2)
    ## ax[0].imshow(X_cp.T, aspect = X_cp.shape[0] / X_cp.shape[1])
    ## ax[1].imshow(X_season.T, aspect = X_season.shape[0] / X_season.shape[1])
    ## print(s)
    return (X_cp_all, X_season_all, s), (X_cp, X_season, s)


def generate_gam(
    X_changepoint, X_seasonality, s_changepoint_times, n_t, n_holdout, ts_mean
):
    @tfd.JointDistributionCoroutine
    def gam():
        y_season, y_trend, noise_sigma = yield from gam_trend_seasonality(
            X_changepoint, X_seasonality, s_changepoint_times, n_t, ts_mean
        )
        y_hat = y_season + y_trend
        if n_holdout > 0:
            y_hat = y_hat[..., :-n_holdout]
        y_obs = yield tfd.Independent(  # pylint: disable=unused-variable
            tfd.Normal(y_hat, noise_sigma[..., None]),
            reinterpreted_batch_ndims=1,
            name="y_obs",
        )

    return gam


def generate_gam_ar1likelihood(
    X_changepoint, X_seasonality, s_changepoint_times, n_t, n_holdout, ts_mean
):
    @tfd.JointDistributionCoroutine
    def gam():
        y_season, y_trend, noise_sigma = yield from gam_trend_seasonality(
            X_changepoint, X_seasonality, s_changepoint_times, n_t, ts_mean
        )
        y_hat = y_season + y_trend
        if n_holdout > 0:
            y_hat = y_hat[..., :-n_holdout]
        rho = yield Root(tfd.Uniform(-1.0, 1.0, name="rho"))

        def ar1_func(y_fluctuation):
            mu_t = (
                tf.concat(
                    [tf.zeros_like(y_fluctuation[..., :1]), y_fluctuation[..., :-1]],
                    axis=-1,
                )
                * rho[..., None]
                + y_hat
            )
            return tfd.Independent(
                tfd.Normal(loc=mu_t, scale=noise_sigma[..., None]),
                reinterpreted_batch_ndims=1,
            )

        # pylint: disable=unused-variable
        y_obs = yield tfd.Autoregressive(
            distribution_fn=ar1_func,
            sample0=tf.zeros_like(y_hat),
            num_steps=1,
            name="y_obs",
        )

    return gam




def generate_gam_ar1(
    X_changepoint, X_seasonality, s_changepoint_times, n_t, n_holdout, ts_mean
):
    @tfd.JointDistributionCoroutine
    def gam():
        y_season, y_trend, noise_sigma = yield from gam_trend_seasonality(
            X_changepoint, X_seasonality, s_changepoint_times, n_t, ts_mean
        )

        ar_sigma = yield Root(tfd.HalfNormal(.5, name="ar_sigma"))
        rho = yield Root(tfd.Uniform(-1.0, 1.0, name="rho"))
        def ar1_func(y_fluctuation):
            mu_t = (
                tf.concat(
                    [tf.zeros_like(y_fluctuation[..., :1]), y_fluctuation[..., :-1]],
                    axis=-1,
                )
                * rho[..., None]
            )
            return tfd.Independent(
                tfd.Normal(loc=mu_t, scale=ar_sigma[..., None]),
                reinterpreted_batch_ndims=1,
            )
        y_ar_error = yield Root(tfd.Autoregressive(
            distribution_fn=ar1_func,
            sample0=tf.zeros_like(y_trend),
            num_steps=y_trend.shape[-1],
            name="y_ar_error",
        ))

        y_hat = y_season + y_trend + y_ar_error
        if n_holdout > 0:
            y_hat = y_hat[..., :-n_holdout]

        # pylint: disable=unused-variable
        y_obs = yield tfd.Independent(
            tfd.Normal(y_hat, noise_sigma[..., None]),
            reinterpreted_batch_ndims=1,
            name="y_obs",
        )

    return gam

