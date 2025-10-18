import numpy as np
from scipy import integrate
from utils import lewis_formula_otm_price
from black import black_impvol


def coef_D(u, tau, params):
    """
    D coefficient in the Heston characteristic exponent.

    Parameters
    ----------
    u : complex or array_like
        Fourier variable (may be complex).
    tau : float
        Time to maturity.
    params : dict
        Model parameters with keys 'kappa', 'rho', 'eta'.

    Returns
    -------
    complex or ndarray
        Value of D(u, tau).
    """
    kappa = params["kappa"]
    rho = params["rho"]
    eta = params["eta"]
    al = -u * u / 2 - 1j * u / 2
    bet = kappa - rho * eta * 1j * u
    gam = eta**2 / 2
    d = np.sqrt(bet * bet - 4 * al * gam)
    rp = (bet + d) / (2 * gam)
    rm = (bet - d) / (2 * gam)
    g = rm / rp
    D = rm * (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
    return D


def coef_C(u, tau, params):
    """
    C coefficient in the Heston characteristic exponent.

    Parameters
    ----------
    u : complex or array_like
        Fourier variable (may be complex).
    tau : float
        Time to maturity.
    params : dict
        Model parameters with keys 'kappa', 'rho', 'eta'.

    Returns
    -------
    complex or ndarray
        Value of C(u, tau).
    """
    kappa = params["kappa"]
    rho = params["rho"]
    eta = params["eta"]
    al = -u * u / 2 - 1j * u / 2
    bet = kappa - rho * eta * 1j * u
    gam = eta**2 / 2
    d = np.sqrt(bet * bet - 4 * al * gam)
    rp = (bet + d) / (2 * gam)
    rm = (bet - d) / (2 * gam)
    g = rm / rp
    C = kappa * (rm * tau - 2 / eta**2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))
    return C


def charfunc_heston(u, tau, params):
    """
    Characteristic function of log-returns under Heston.

    Parameters
    ----------
    u : complex or array_like
        Argument of the characteristic function.
    tau : float
        Time to maturity.
    params : dict
        Model parameters with keys 'kappa', 'rho', 'eta', 'vbar', 'v'.

    Returns
    -------
    complex or ndarray
        E[exp(i*u*X_tau)] where X_tau = log(S_tau/S_0).
    """
    return np.exp(
        coef_C(u, tau, params) * params["vbar"] + coef_D(u, tau, params) * params["v0"]
    )


def ssr_heston_charfunc(tau, params):
    """
    Skew-Stickiness Ratio (SSR) via Heston characteristic function integrals.

    Parameters
    ----------
    tau : float
        Time to maturity.
    params : dict
        Model parameters with keys 'kappa', 'rho', 'eta', 'vbar', 'v0'.

    Returns
    -------
    float
        SSR value (ratio of two integrals).
    """
    rho = params["rho"]
    eta = params["eta"]
    vbar = params["vbar"]
    v0 = params["v0"]

    def func_C(a):
        return coef_C(a - 0.5j, tau=tau, params=params)

    def func_D(a):
        return coef_D(a - 0.5j, tau=tau, params=params)

    def num_integrand(a):
        return np.real(
            rho
            * eta
            * func_D(a)
            * np.exp(func_D(a) * v0 + func_C(a) * vbar)
            / (a**2 + 0.25)
        )

    def denom_integrand(a):
        return np.imag(np.exp(func_D(a) * v0 + func_C(a) * vbar) * a / (a**2 + 0.25))

    num = integrate.quad(num_integrand, 0, np.inf)[0]
    denom = integrate.quad(denom_integrand, 0, np.inf)[0]
    return num / denom


def ssr_heston_forest(tau, params):
    """
    Forest expansion approximation of SSR for Heston when kappa=0 and vbar=v0.
    See Section 5.3.1 in [Friz, Gatheral, "Computing the SSR", QF, 2025]

    Parameters
    ----------
    tau : float
        Time to maturity (must be > 0).
    params : dict
        Model parameters with keys 'kappa', 'rho', 'eta', 'vbar', 'v0'.

    Returns
    -------
    float
        Approximate SSR using the forest expansion up to O(tau).
    """
    rho = params["rho"]
    eta = params["eta"]
    vbar = params["vbar"]
    v0 = params["v0"]
    kappa = params["kappa"]

    if vbar != v0:
        raise ValueError("vbar must be equal to v0 for this approximation.")

    if np.any(tau <= 0):
        raise ValueError("tau must be positive.")

    if kappa != 0:
        raise ValueError("kappa must be equal to 0 for this approximation.")

    # Numerator expansion up to O(tau)
    num = (
        1
        + (1 / 8) * eta * rho * tau
        + (1 / 24) * (eta**2) * tau / v0
        - (1 / 96) * (rho**2) * (eta**2) * tau / v0
    )

    # Denominator expansion up to O(tau)
    denom = (
        1
        - (1 / 24) * rho * eta * tau
        + (1 / 8) * (eta**2) * tau / v0
        - (3 / 32) * (rho**2) * (eta**2) * tau / v0
    )

    return 2.0 * num / denom


def phi_heston(u, tau, params):
    """
    Compute the characteristic function of the Heston model E[exp(i * u * X_tau)]
    where X_tau = log(S_tau/S_0) at u for time tau.

    Parameters
    ----------
    u : float or array_like
        Points at which to evaluate the characteristic function
    tau : float
        Time to maturity
    params : dict
        Dictionary containing model parameters:
        - kappa: mean reversion rate
        - rho: correlation between asset and variance
        - eta: volatility of variance
        - vbar: long-term variance
        - v0: initial variance

    Returns
    -------
    complex
        Value of the characteristic function
    """
    kappa = params["kappa"]
    rho = params["rho"]
    eta = params["eta"]
    vbar = params["vbar"]
    v0 = params["v0"]

    al = -u * u / 2 - 1j * u / 2
    bet = kappa - rho * eta * 1j * u
    gam = eta**2 / 2
    d = np.sqrt(bet * bet - 4 * al * gam)
    rp = (bet + d) / (2 * gam)
    rm = (bet - d) / (2 * gam)
    g = rm / rp
    D = rm * (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
    C = kappa * (rm * tau - 2 / eta**2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))
    return np.exp(C * vbar + D * v0)


def impvol_heston_charfunc(k, tau, params):
    """
    Calculate implied volatility in the Heston model using the characteristic function.

    Parameters
    ----------
    k : array_like
        Log strike k = log(K/F)
    tau : float
        Time to maturity
    params : dict
        Model parameters

    Returns
    -------
    array_like
        Black implied volatility
    """
    k = np.atleast_1d(np.asarray(k))
    otm_price = lewis_formula_otm_price(
        lambda u, tau: phi_heston(u=u, tau=tau, params=params),
        k=k,
        tau=tau,
    )
    opttype = 2 * (k > 0) - 1  # otm options
    impvol = black_impvol(K=np.exp(k), T=tau, F=1, value=otm_price, opttype=opttype)
    return impvol


def simulate_paths_qe_scheme(
    T, params, n_disc, n_paths, psi_c=1.5, seed=None, eps=1e-14
):
    """
    Simulate Heston model paths using Andersen's QE discretization scheme.

    Parameters
    ----------
    T : float
        Time to maturity.
    params : dict
        Model parameters: 'S0', 'v0', 'kappa', 'vbar', 'eta', 'rho'.
    n_disc : int
        Number of discretization steps.
    n_paths : int
        Number of Monte Carlo paths.
    psi_c : float, default 1.5
        Critical value for quadratic/exponential scheme switching.
    seed : int, optional
        Random seed for reproducibility.
    eps : float, default 1e-14
        Floor to keep conditional mean/variance and v_t non-negative.

    Returns
    -------
    S : ndarray, shape (n_disc + 1, n_paths)
        Stock price paths.
    v : ndarray, shape (n_disc + 1, n_paths)
        Variance paths.
    """
    if seed is not None:
        np.random.seed(seed)

    eta = params["eta"]
    kappa = params["kappa"]
    vbar = params["vbar"]
    rho = params["rho"]

    logS_qe = np.zeros((n_disc + 1, n_paths), dtype=float)
    v_qe = np.zeros((n_disc + 1, n_paths), dtype=float)
    logS_qe[0, :] = np.log(params["S0"])
    v_qe[0, :] = params["v0"]

    ts = np.linspace(0.0, T, n_disc + 1)
    dt = ts[1] - ts[0]
    edt = np.exp(-kappa * dt)

    for i in range(n_disc):
        # conditional mean and variance
        m = (v_qe[i, :] - vbar) * edt + vbar
        m = np.maximum(m, eps)  # ensure positivity
        s2 = (eta**2 / kappa) * (
            edt * (1 - edt) * (v_qe[i, :] - vbar) + vbar / 2 * (1 - edt**2)
        )
        # compute relative variance
        psi = s2 / m**2

        # Regime 1: Quadratic form
        mask_quad = psi <= psi_c
        if np.any(mask_quad):
            psi_quad = psi[mask_quad]
            Z_quad = np.random.normal(size=mask_quad.sum())
            b2 = (2.0 + 2.0 * np.sqrt(1.0 - psi_quad / 2.0) - psi_quad) / psi_quad
            a = m[mask_quad] / (1.0 + b2)
            v_qe[i + 1, mask_quad] = a * (b2**0.5 + Z_quad) ** 2

        # Regime 2: Exponential form
        mask_exp = ~mask_quad
        if np.any(mask_exp):
            psi_exp = psi[mask_exp]
            # clip p to [0, 1)
            p = np.clip((psi_exp - 1.0) / (psi_exp + 1.0), 0.0, 1.0 - 1e-15)
            beta = m[mask_exp] * (psi_exp + 1.0) / 2.0

            U_exp = np.random.uniform(0.0, 1.0, size=mask_exp.sum())
            alive = U_exp > p  # if False -> atom at zero

            # start at zero
            v_qe[i + 1, mask_exp] = 0.0
            if np.any(alive):
                U_exp_new = np.random.uniform(0.0, 1.0, size=alive.sum())
                U_exp_new = np.clip(U_exp_new, eps, 1.0)
                idx_exp = np.where(mask_exp)[0]
                alive_idx = idx_exp[alive]
                v_qe[i + 1, alive_idx] = -beta[alive] * np.log(U_exp_new)

        v_qe[i + 1, :] = np.maximum(v_qe[i + 1, :], eps)  # ensure positivity

        # trapezoidal rule for integrated variance
        int_v_trap_i = 0.5 * (v_qe[i, :] + v_qe[i + 1, :]) * dt
        int_v_trap_i = np.maximum(int_v_trap_i, 0.0)

        logS_qe[i + 1, :] = (
            logS_qe[i, :]
            - 0.5 * int_v_trap_i
            + rho
            * (v_qe[i + 1, :] - v_qe[i, :] - kappa * vbar * dt + kappa * int_v_trap_i)
            / eta
            + np.sqrt(np.maximum((1.0 - rho**2) * int_v_trap_i, 0.0))
            * np.random.normal(size=n_paths)
        )

    return np.exp(logS_qe), v_qe
