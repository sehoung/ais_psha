import shapely.geometry as geom
import numpy as np

### FUNCTION FOR GROUND MOTION ###
def calc_meanlnSA(M, R, gmmid = 1000):
    if gmmid == 1000:
        lnmean, lnsigma = sadigh1997(M,R, T='PGA')
    return lnmean, lnsigma
####################

def sadigh1997(M, R, T='PGA'):
    # Initialize coefficients arrays
    C1 = np.zeros_like(M)
    C2 = np.zeros_like(M)
    C3 = np.zeros_like(M)
    C4 = -2.100 * np.ones_like(M)
    C5 = np.zeros_like(M)
    C6 = np.zeros_like(M)
    C7 = np.zeros_like(M)
    if T=='PGA':       
        # Conditions based on M values
        condition = M <= 6.5
        C1[condition] = -0.624
        C2[condition] = 1.0
        C3[condition] = 0.000
        C5[condition] = 1.29649
        C6[condition] = 0.250

        condition = M >= 6.5
        C1[condition] = -1.274
        C2[condition] = 1.1
        C3[condition] = 0.000
        C5[condition] = -0.48451
        C6[condition] = 0.524

        # Calculate lnmean for all M and R
        Rrup = R
        lnmean = C1 + C2 * M + C3 * (8.5 * M)**2.5 + C4 * np.log(Rrup + np.exp(C5 + C6 * M)) + C7 * np.log(Rrup + 2)

        # Calculate lnsigma for all M
        lnsigma = np.where(M < 7.21, 1.39 - 0.14 * M, 0.38)

        return lnmean, lnsigma
        
