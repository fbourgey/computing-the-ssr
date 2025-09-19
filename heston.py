import numpy as np
from scipy import integrate


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
        coef_C(u, tau, params) * params["vbar"] + coef_D(u, tau, params) * params["v"]
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
