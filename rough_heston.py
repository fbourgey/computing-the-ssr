import numpy as np
from scipy import integrate
from scipy import special as sp
from black import black_impvol
from utils import gauss_legendre, lewis_formula_otm_price
import pade
from heston import ssr_heston_charfunc


def charfunc_rheston(u, tau, params, xi_curve, n_pade: int = 2, n_quad: int = 30):
    """
    Padé-based rational approximation of the rough Heston characteristic function.

    Uses a Padé approximant for the integrand and Gauss-Legendre (or quad_vec)
    integration to compute E[exp(i u X_tau)] for given u and tau.

    Parameters
    ----------
    u : float or ndarray
        Fourier argument(s) for the characteristic function.
    tau : float or ndarray
        Time(s) to maturity.
    params : dict
        Model parameters for rough Heston (e.g., nu, H, lbd, rho).
    xi_curve : callable
        Forward variance curve function xi_curve(t).
    n_pade : int, optional
        Order of Padé approximation (default: 2).
    n_quad : int, optional
        Number of quadrature points for integration (default: 30).

    Returns
    -------
    ndarray
        Characteristic function values for each tau.
    """
    if n_pade not in [2, 3, 4, 5, 6]:
        raise ValueError("Invalid Padé order. Must be 2, 3, 4, 5, or 6.")

    tau = np.atleast_1d(np.asarray(tau))
    if n_quad > 0:
        # Gauss quadrature
        x_le, w_le = gauss_legendre(0, 1, n_quad)
        log_charfunc = (
            w_le[None, :]
            * xi_curve(tau[:, None] * (1 - x_le[None, :]))
            * pade.g(
                a=u,
                t=tau * x_le[None, :],
                params=params,
                n_pade=n_pade,
            )
        )
        log_charfunc = np.sum(log_charfunc, axis=1)
    else:
        # Using scipy.quad_vec() is longer.
        log_charfunc = integrate.quad_vec(
            lambda t: xi_curve(tau * (1 - t))
            * pade.g(a=u, t=tau * t, params=params, n_pade=n_pade),
            0,
            1,
            epsrel=1e-10,
            limit=1000,
        )[0]
    charfunc = np.exp(tau * log_charfunc)
    return charfunc


def ssr_rheston_charfunc(tau, params, xi_curve, n_pade: int = 2, n_quad: int = 30):
    rho = params["rho"]
    nu = params["nu"]

    tau = np.atleast_1d(np.asarray(tau))
    ssr = np.zeros_like(tau)

    H = params["H"]
    if H == 0.5:
        # Heston case (H=1/2) - convert to Heston variables

        # implicity assumes v0=vbar (flat forward variance curve)
        # check that xi_curve is flat
        us = np.linspace(0.0, 2.0, 100)
        vs = xi_curve(us)
        assert np.allclose(vs, vs[0])

        params = params.copy()
        params["v0"] = xi_curve(0.0)
        params["vbar"] = xi_curve(0.0)
        params["eta"] = nu
        params["kappa"] = params["lbd"]
        for i, tau_i in enumerate(tau):
            ssr[i] = ssr_heston_charfunc(tau_i, params)
    else:
        for i, tau_i in enumerate(tau):

            def _cf(a):
                return charfunc_rheston(
                    u=a - 0.5j,
                    tau=tau_i,
                    params=params,
                    xi_curve=xi_curve,
                    n_pade=n_pade,
                    n_quad=n_quad,
                )

            def integrand_num(a):
                return np.real(
                    rho
                    * nu
                    * pade._h_pade(tau=tau_i, a=a - 0.5j, params=params, n_pade=n_pade)
                    * _cf(a=a)
                    / (a**2 + 0.25)
                )

            def integrand_denom(a):
                return np.imag(_cf(a=a) * a / (a**2 + 0.25))

            num = integrate.quad(integrand_num, 0, np.inf)[0]
            denom = integrate.quad(integrand_denom, 0, np.inf)[0]
            ssr[i] = num / denom

    return ssr


def ssr_forest_expansion(tau, params, xi0_flat, explicit=False):
    """
    Compute SSR via the Forest expansion.

    Parameters
    ----------
    tau : float or np.ndarray
        Maturity horizon(s) (T > 0).
    params : dict
        Model parameters with keys 'H', 'rho', 'nu'.
    xi0_flat : float or ndarray
        Flat initial forward variance xi0.
    explicit : bool, optional
        If True use explicit coefficient formulas (default False).

    Returns
    -------
    float or np.ndarray
        SSR value(s) with the same shape as tau.

    Notes
    -----
    This implementation assumes a flat initial forward variance curve and zero speed
    of mean reversion (lbd = 0).

    References
    ----------
    Friz, Gatheral (2025), "Computing the SSR". Proposition 5.8.
    """
    H = params["H"]
    rho = params["rho"]
    nu = params["nu"]
    xi0 = xi0_flat
    alpha_h = H + 0.5
    xi0_tau = xi0 * tau

    # --- gamma function shorthands ---
    gamma1_alpha = sp.gamma(1 + alpha_h)
    gamma2_alpha = sp.gamma(2 + alpha_h)
    gamma1_2alpha = sp.gamma(1 + 2 * alpha_h)
    gamma2_2alpha = sp.gamma(2 + 2 * alpha_h)
    gamma1_3alpha = sp.gamma(1 + 3 * alpha_h)
    gamma2_3alpha = sp.gamma(2 + 3 * alpha_h)

    if explicit:
        c0 = -gamma1_2alpha / (4.0 * gamma1_alpha**2 * gamma1_3alpha) + 3.0 / (
            8.0 * (2.0 * alpha_h + 1.0) * gamma1_alpha**3
        )
        c1 = (
            -1.0 / gamma1_3alpha
            + 3.0 / (2.0 * gamma2_alpha * gamma1_2alpha)
            + 3.0 / (2.0 * gamma1_alpha * gamma2_2alpha)
            - 15.0 / (8.0 * gamma1_alpha * gamma2_alpha**2)
        )
        d0 = (
            15.0 / (8.0 * (2.0 * alpha_h + 1.0) * gamma1_alpha**2 * gamma2_alpha)
            - 3.0 * gamma1_2alpha / (4.0 * gamma1_alpha**2 * gamma2_3alpha)
            - 3.0 / (2.0 * gamma1_alpha * gamma1_2alpha * (3.0 * alpha_h + 1.0))
        )
        d1 = (
            15.0 / (2.0 * gamma2_alpha * gamma2_2alpha)
            - 3.0 / gamma2_3alpha
            - 35.0 / (8.0 * gamma2_alpha**3)
        )

        numerator = (
            rho * xi0 * nu * tau ** (alpha_h + 1.0) / gamma1_alpha
            + rho**2
            * xi0
            * nu**2
            * tau ** (2.0 * alpha_h + 1.0)
            * (1.0 / (2.0 * gamma1_2alpha) - 1.0 / (4.0 * gamma1_alpha * gamma2_alpha))
            + rho * nu**3 * tau ** (3.0 * alpha_h) * (c0 + rho**2 * c1)
        )
        denominator = (
            xi0 * rho * nu * tau ** (1.0 + alpha_h) / gamma2_alpha
            + xi0
            * rho**2
            * nu**2
            * tau ** (2.0 * alpha_h + 1.0)
            * (1.0 / gamma2_2alpha - 3.0 / (4.0 * gamma2_alpha**2))
            + rho * nu**3 * tau ** (3.0 * alpha_h) * (d0 + rho**2 * d1)
        )
    else:
        # --- kernel terms ---
        o_g = rho * nu * xi0 * tau ** (alpha_h + 1) / gamma2_alpha
        o_o = (
            nu**2
            * xi0
            * tau ** (2 * alpha_h + 1)
            / (gamma1_alpha**2 * (2 * alpha_h + 1))
        )
        o_g_g = rho**2 * nu**2 * xi0 * tau ** (2 * alpha_h + 1) / gamma2_2alpha
        g_o_o = (
            rho
            * nu**3
            * xi0
            * tau ** (3 * alpha_h + 1)
            / (gamma1_alpha * gamma1_2alpha * (3 * alpha_h + 1))
        )
        o_o_g = (
            rho
            * nu**3
            * gamma1_2alpha
            * xi0
            * tau ** (3 * alpha_h + 1)
            / (gamma1_alpha**2 * gamma2_3alpha)
        )
        o_g_g_g = rho**3 * nu**3 * xi0 * tau ** (3 * alpha_h + 1) / gamma2_3alpha

        # --- derivatives ---
        dxi_o = nu * tau**alpha_h / gamma1_alpha
        dxi_o_g = rho * nu**2 * tau ** (2 * alpha_h) / gamma1_2alpha
        dxi_o_o = (
            nu**3
            * gamma1_2alpha
            * tau ** (3 * alpha_h)
            / (gamma1_alpha**2 * gamma1_3alpha)
        )
        dxi_o_g_g = rho**2 * nu**3 * tau ** (3 * alpha_h) / gamma1_3alpha

        # --- SSR formula ---
        numerator = (
            dxi_o
            + dxi_o_g / 2.0
            - dxi_o_o / (4.0 * xi0_tau)
            - dxi_o_g_g / xi0_tau
            - o_g * dxi_o / (4 * xi0_tau)
            + 3 * o_g * dxi_o_g / (2 * xi0_tau**2)
            + 3 * dxi_o * o_o / (8 * xi0_tau**2)
            + 3 * dxi_o * o_g_g / (2 * xi0_tau**2)
            - 15 * dxi_o * o_g**2 / (8 * xi0_tau**3)
        )
        numerator *= rho * xi0_tau

        denominator = (
            o_g
            + o_g_g
            - 3 * o_g**2 / (4 * xi0_tau)
            - 105 * o_g**3 / (24 * xi0_tau**3)
            + 15 * o_o * o_g / (8 * xi0_tau**2)
            + 15 * o_g_g * o_g / (2 * xi0_tau**2)
            - 3 * o_o_g / (4 * xi0_tau)
            - 3 * g_o_o / (2 * xi0_tau)
            - 3 * o_g_g_g / xi0_tau
        )

    return numerator / denominator


def impvol_rheston_rational(k, tau, params, xi, n_pade: int, n_quad: int = 40):
    """
    Calculate implied volatility in the rough Heston model using a rational
    approximation of the characteristic function.

    Parameters
    ----------
    k : float
        Log strike
    tau : float
        Time to maturity
    params : dict
        Model parameters
    xi : ndarray
        Volatility curve values
    n : int
        Order of rational approximation

    Returns
    -------
    float
        Black implied volatility
    """
    if n_pade not in [2, 3, 4, 5, 6]:
        raise ValueError("Invalid Padé order. Must be 2, 3, 4, 5, or 6.")

    otm_price = lewis_formula_otm_price(
        lambda u, tau: charfunc_rheston(
            u=u, tau=tau, params=params, xi_curve=xi, n_pade=n_pade, n_quad=n_quad
        ),
        k=k,
        tau=tau,
    )
    opttype = 2 * (k > 0) - 1  # otm options
    impvol = black_impvol(K=np.exp(k), T=tau, F=1, value=otm_price, opttype=opttype)
    return impvol
