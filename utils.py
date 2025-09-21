import numpy as np
from scipy.integrate import quad_vec


def gauss_legendre(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Legendre quadrature points and weights on the interval [a, b].

    Parameters
    ----------
    a : float
        Lower bound of the integration interval.
    b : float
        Upper bound of the integration interval.
    n : int
        Number of quadrature points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two 1-D arrays:
        - Quadrature points on [a, b].
        - Quadrature weights on [a, b].
    """
    knots, weights = np.polynomial.legendre.leggauss(n)
    knots_a_b = 0.5 * (b - a) * knots + 0.5 * (b + a)
    weights_a_b = 0.5 * (b - a) * weights
    return knots_a_b, weights_a_b


def non_uniform_grid(a: float, b: float, n: int, power: float = 3.0) -> np.ndarray:
    """
    Create a non-uniform grid on the interval [a, b] with clustering towards 'a'.

    Parameters
    ----------
    a : float
        Start of the interval.
    b : float
        End of the interval.
    n : int
        Number of grid points.
    power : float, optional
        Power for clustering (default is 3.0). Higher values cluster more towards 'a'.

    Returns
    -------
    np.ndarray
        1-D array of grid points on [a, b].
    """
    uniform_grid = np.linspace(0, 1, n)
    non_uniform_grid = a + (b - a) * uniform_grid**power
    return non_uniform_grid


def lewis_formula_otm_price(phi, k, tau):
    """
    Compute the OTM (Out-of-The-Money) option price using the Lewis formula.

    The Lewis formula is used to price European options by applying Fourier transform
    methods. It calculates the price of OTM options given the characteristic function
    of the log price.

    Parameters
    ----------
    phi : callable
        The characteristic function of the log price process.
        Should take two arguments: complex number u and time to maturity tau.
    k : float or array_like
        Log strike price(s). k = log(K/S) where K is strike and S is spot price.
    tau : float or array_like
        Time(s) to maturity in years.

    Returns
    -------
    ndarray
        OTM option prices. For k < 0, returns put prices; for k >= 0, returns
        call prices.

    Notes
    -----
    The formula uses the following representation:
    For k >= 0 (calls): C(k,T) = 1/π ∫[0,∞] Re[e^(-iuk)φ(u-i/2,τ)/(u^2+1/4)]du
    For k < 0 (puts): P(k,T) = e^k - 1/π ∫[0,∞] Re[e^(-iuk)φ(u-i/2,τ)/(u^2+1/4)]du

    References
    ----------
    Lewis, A. L. (2000). Option Valuation under Stochastic Volatility.
    """
    k = np.atleast_1d(np.asarray(k))
    tau = np.atleast_1d(np.asarray(tau))

    def integrand(u):
        return np.real(np.exp(-1j * u * k) * phi(u - 1j / 2, tau) / (u**2 + 1 / 4))

    k_minus = k * (k < 0)

    integral, _ = quad_vec(integrand, 0, np.inf, epsrel=1e-10, limit=1000)
    result = np.exp(k_minus) - np.exp(k / 2) / np.pi * integral

    return result
