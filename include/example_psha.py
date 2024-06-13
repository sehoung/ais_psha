import numpy as np
from scipy import stats
from include.gmm import calc_meanlnSA
from include.fault_distance_pdf import calc_pdf_dist_fault

def PSHA_area1(X, arg=[]):
    M, D, Z, eps = X[:, 0], X[:, 1], X[:,2], X[:,3]
    a=arg[0]

    rate = 0.0395
    bval=0.9
    mmin = 5.0
    mmax = 6.5
    rate = 0.0395

    # calculate pdf
    fM = np.where((M>=mmin) & (M<=mmax), (bval) * np.log(10)*10**(-(bval) * (M - mmin)) / (1 - 10**(-bval * (mmax - mmin))), 0 )
    fD = np.where( (D>0) & (D<100), 2*D/100**2, 0)
    fZ = stats.uniform.pdf(Z, loc=5, scale=5)
    feps = stats.norm.pdf(eps)

    # Simulating ground motion intensity
    R = np.sqrt( D*D + Z*Z )
    meanlnSA, sigmalnSA = calc_meanlnSA(M, R)
    lnGM = meanlnSA + sigmalnSA * eps
    I = np.where(lnGM > np.log(a), 1., 0)
    return rate * I * fM * fD * fZ * feps