import numpy as np

class Charcteristic:
    def __init__(self, bval=1, mmin=5.0, mmax=7.0, dm=0.1, dMchar = 0.5, sr=1.0):
        self.bval = bval
        self.mmin = mmin
        self.mmax = mmax
        self.dm = dm
        self.dMchar = dMchar
        self.sr = sr

    def calc_yc1985_pdf_pmf(self, M, A):
        
        Nm0, Nmc = self.calc_yc1985_Nm0_Nmc(self, A)
        range_exp = (M >= self.mmin) & (M <= self.mmax-self.dMchar)
        range_char = (M > self.mmax-self.dMchar) & (M <= self.mmax)
        range_invalid = (M <  self.mmin) | (M > self.mmax)

        # Initialize Mpdf with the same shape as M
        #Mpdf = np.empty_like(M)
        #Mpdf[range_exp] = formula1[range_exp]
        #Mpdf[range_char] = formula2[rhange_char]

        nM = np.empty_like(M)
        nM[range_exp] = (Nm0-Nmc) * self.bval * np.log(10) * 10**(-self.bval * (M[range_exp] - self.mmin)) / (1 - 10**(-self.bval * (self.mmax - self.mmin)))
        nM[range_char] = np.zeros_like(M[range_char]) + Nmc/self.dMchar
        nM[range_invalid] = 0
    
        yc1985_pdf = nM / Nm0
        yc1985_pdf #####
        yc1985_pmf = yc1985_pdf * self.dm
        Normfactor = np.sum(yc1985_pmf)

        yc1985_pdf = yc1985_pdf / Normfactor
        yc1985_pmf = yc1985_pmf / Normfactor


        return yc1985_pdf, yc1985_pmf, Nm0

    def calc_yc1985_Nm0_Nmc(self,  A):
        c=1.5
        M0rate = self.SR_to_M0RATE(self.sr, A)
        beta = self.bval * np.log(10)
        
        M0u = self.M_to_M0(self.mmax)    
        AA = np.exp( -beta*( self.mmax - self.mmin - 0.5 ) )
        BB = self.bval * 10**(-c/2.) / (c-self.bval)
        CC = self.bval * np.exp(beta) * ( 1 - 10**(-c/2.) ) / c

        X = M0rate * (1-AA) / ( AA * M0u * ( BB + CC )  ) ## X = Nm0 - Nmc (Total number of ea in exp range)
        Nmc = beta * X * AA * np.exp(beta) / ( 2 * ( 1 - AA ) ) ## Total N of eq in characteristic range
        Nm0 = X + Nmc ## Total N of eq of the whole range

        return Nm0, Nmc

    def M_to_M0(M): # M0 = Nm
        M0 = 10**(9.05 + 1.5*M)
        return M0

    def M0_to_M(M0): # M0 = Nm
        M = 1/1.5 * (np.log10(M0) - 9.05)
        return M

    def SR_to_M0RATE(self, area):
        """
        Args:
            slip_rate (numpy array): mm/yr
            area (numpy array): km2

        Returns:
            numpy array: M0rate in Nm/yr
        """
        shear_modulus = 3*10**10 ## N/m2, == 3*10**11 dyne/cm2
        m0rate = shear_modulus * area * self.sr * 1000
        return m0rate 
    ##################



class DoubleTruncateExp:
    def __init__(self, rate=-1, sr=-1, bval=1, mmin=5.0, mmax=7.0, dm=0.1):
        if rate*sr >0 :
            raise ValueError("Only one rate parameter either \'rate\' or \'sr\' (slip rate) should be specified.")
        self.rate = rate
        self.bval = bval
        self.mmin = mmin
        self.mmax = mmax
        self.dm = dm

    def pmf(self, M):
        # Create a boolean mask for values of M within the specified range
        valid_range = (M >= self.mmin) & (M <= self.mmax)

        # Initialize GRpdf and GRpmf arrays with zeros
        GRpmf = np.zeros(M.shape)

        # Calculate GRpdf and GRpmf only for valid M values
        GRpmf[valid_range] = self.cdf(M[valid_range] + self.dm / 2) - self.cdf(M[valid_range] - self.dm / 2)
        return GRpmf
    
    def pdf(self, M):
        valid_range = (M >= self.mmin) & (M <= self.mmax)
        GRpdf = np.zeros(M.shape)

        A = (1 - 10**(-self.bval * (self.mmax - self.mmin)))
        GRpdf[valid_range] = ( self.bval * np.log(10) * 10**(-self.bval * (M[valid_range] - self.mmin))) / A
        
        return GRpdf
    
    def cdf(self, M):
        cdf = np.zeros(M.shape)
        cdf[M < self.mmin] = 0
        cdf[M > self.mmax] = 1

        within_range = (M >= self.mmin) & (M <= self.mmax)
        A = (1 - 10**(-self.bval * (self.mmax - self.mmin)))
        cdf[within_range] = (1 - 10**(-self.bval * (M[within_range] - self.mmin))) / A
        return cdf

    def sample(self, Nsmpl=1):
        A = (1 - 10**(-self.bval * (self.mmax - self.mmin)))
        cdf = np.random.uniform(size=Nsmpl)
        M = np.log10(1 - A * cdf) / (-self.bval) + self.mmin
        return M
    
    def calc_rate(self):
        return self.rate

# Example usage
if __name__ == "__main__":
    model = DoubleTruncateExp(bval=1, mmin=5.0, mmax=7.0)
    M = np.linspace(5, 7, 100)
    dm = 0.1

    GRpdf, GRpmf = model.calc_pdf_pmf(M, dm)
    print("PDF:", GRpdf)
    print("PMF:", GRpmf)

    samples = model.sample(Nsmpl=10)
    print("Samples:", samples)
