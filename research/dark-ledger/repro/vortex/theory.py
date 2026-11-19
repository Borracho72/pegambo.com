"""Berry & Dennis 2000 Eqs. (2.26), (4.46), (4.47), (5.7), (5.9)."""
import numpy as np
from scipy.integrate import quad
from scipy.special import jv


def _corr(z, cos, sin, pi, sqrt, J, integral):
    """Pair and signed pair correlations, z = k*r. Real component variance = 1."""
    c = J(0, z)
    cp = -J(1, z)
    cpp = (J(2, z) - c)/2
    d = 1/2
    s = (1-c)*(1+c)
    a = d - cp**2/s
    L = (-cpp-c*cp**2/s)/a
    T = -cp/(z*d)
    Y, Z = (L*T)**2, (1-L**2)*(1-T**2)

    def integrand(theta):
        u, v = cos(theta)**2, sin(theta)**2
        return u*((3-Z+2*Y)*u**2 + (3+Z-2*Y)*v*u + 2*Z*v**2)/sqrt(u**2+(1+Z-Y)*v*u+Z*v**2)

    I = integral(integrand, pi/2)
    rho = 2*a/(d*s)
    return rho*I/pi, rho*L*T


def correlations(r):
    """Returns unmarked g, signed gQ, like and unlike partial g; r in wavelengths."""
    if r == 0:
        return .25, -.25, 0, .5
    g, gq = _corr(2*np.pi*r, np.cos, np.sin, np.pi, np.sqrt, jv,
                  lambda f, b: quad(f, 0, b, epsabs=1e-12, epsrel=1e-12)[0])
    return g, gq, g+gq, g-gq


def cubic_coefficient():
    """Kac–Rice small-separation coefficient of g_same, for r/lambda."""
    i = quad(lambda theta: (np.sin(theta)**2+9*np.cos(theta)**2)**1.5,
             0, np.pi, epsabs=1e-12, epsrel=1e-12)[0]
    return np.pi**2*i/216


def high_precision(r):
    """Numerical corroboration of the asymptote with stable cancellation at r << 1."""
    import mpmath as mp
    with mp.workdps(55):
        z = 2*mp.pi*mp.mpf(str(r))
        g, gq = _corr(z, mp.cos, mp.sin, mp.pi, mp.sqrt, mp.besselj,
                      lambda f, b: mp.quad(f,[0,b-z,b-z/10,b] if z<b else [0,b]))
        return g, gq, g+gq, g-gq


if __name__ == "__main__":
    print(f"Exact leading coefficient: {cubic_coefficient():.12g}")
    print("r/lambda, g, gQ, g_same, g_opposite, g_same/(r/lambda)^3")
    for r in (.3, .2, .1, .05, .02, .01, .005, .002, .001):
        g, gq, same, opp = high_precision(r)
        print(f"{r:.3f} {float(g):.12g} {float(gq):.12g} {float(same):.12g} {float(opp):.12g} {float(same/r**3):.12g}")
