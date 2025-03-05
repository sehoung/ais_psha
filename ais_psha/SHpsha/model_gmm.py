import numpy as np
from scipy.stats import norm
class GroundMotionModel:
    def __init__(self, model="sadigh1997", add_lnmed = 0, add_lnsigma = 0, sample_eps = True):
        self.model=model
        self.add_lnmed = add_lnmed
        self.add_lnsigma = add_lnsigma
        self.sample_eps = sample_eps

        if sample_eps == True:
            self.simulate = self.simulate
        else:
            self.simulate = self.simulate_no_eps
    

    def sample(self, Nsmpl=100):
        eps = np.random.normal(size=Nsmpl)
        return eps
    
    def pdf(self, x):
        return norm.pdf(x)
    
    def simulate(self, M, R, eps):
        lnmed, lnsigma = self.calc_mean_sigma(M, R, T='PGA')
        lngm = lnmed + lnsigma*eps
        return np.exp(lngm)

    def simulate_no_eps(self, M, R):
        lnmed, lnsigma = self.calc_mean_sigma(M, R, T='PGA')
        return lnmed, lnsigma

    def calc_mean_sigma(self, M, R, T='PGA', complexity=0):
        match self.model:
            case 'sadigh1997':
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
                    M=np.where(M<0, 0, M)
                    lnmean = C1 + C2 * M + C3 * (8.5 * M)**2.5 + C4 * np.log(Rrup + np.exp(C5 + C6 * M)) + C7 * np.log(Rrup + 2)
                    for i in range(complexity):
                        lnmean = C1 + C2 * M + C3 * (8.5 * M)**2.5 + C4 * np.log(Rrup + np.exp(C5 + C6 * M)) + C7 * np.log(Rrup + 2)
                    
                    # Calculate lnsigma for all M
                    lnsigma = np.where(M < 7.21, 1.39 - 0.14 * M, 0.38)

                    lnmean += self.add_lnmed
                    lnsigma += self.add_lnsigma

                    return lnmean, lnsigma


    #def calc_yc1985_pdf_pmf(self, M, A):

    def calc_mean_sigma_sadigh1997(M, R, T='PGA', complexity=0):
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
            M=np.where(M<0, 0, M)
            lnmean = C1 + C2 * M + C3 * (8.5 * M)**2.5 + C4 * np.log(Rrup + np.exp(C5 + C6 * M)) + C7 * np.log(Rrup + 2)
            for i in range(complexity):
                lnmean = C1 + C2 * M + C3 * (8.5 * M)**2.5 + C4 * np.log(Rrup + np.exp(C5 + C6 * M)) + C7 * np.log(Rrup + 2)
            
            # Calculate lnsigma for all M
            lnsigma = np.where(M < 7.21, 1.39 - 0.14 * M, 0.38)

            return lnmean, lnsigma
            


