import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from SHpsha import GroundMotionModel
#from example3 import PSHA_PEER_Test1_11_general
#from include.example3 import calc_mean_sigma_sadigh1997

class PCE_medianGMM():
    def __init__(self, GMi=0.1, lnmed=0, lnsigma=0.5, sigma_mu=0.2, max_order = 6, Nxi=1000):

        self.GMi = GMi
        self.lnGMi = np.log(GMi)
        self.lnmed = lnmed
        self.lnsigma = lnsigma
        self.sigma_mu = sigma_mu
        self.max_order = max_order
        self.Nmarginal = len(lnmed)

        self.a = -self.sigma_mu**2 / (2*self.lnsigma**2) - 1/2
        self.b = (self.lnGMi-self.lnmed) * self.sigma_mu / self.lnsigma**2
        self.c = -(self.lnGMi-self.lnmed)**2 / (2 * self.lnsigma**2)

        self.alpha = self.sigma_mu / (2*self.lnsigma*np.sqrt(np.pi)) * np.exp( self.c - self.b**2/(4*self.a)  )

        self.Nxi = Nxi
        
    def hermite_poly(self, x, order=0):
        if np.isscalar(x):  # Check if the input is a float
            x=np.array([x])

        Nsmpl = len(x)
        match order:
            case 0:
                return np.full(Nsmpl, 1)
            case 1:
                return x
            case 2:
                return x**2 - 1
            case 3:
                return x**3 - 3*x
            case 4:
                return x**4 - 6*x**2 + 3
            case 5:
                return x**5 - 10*x**3 + 15*x
            case 6:
                return x**6 - 15*x**4 + 45*x**2 - 15
            case _:
                raise ValueError("order should be 0, 1, 2, ..., 6")


    def coeff_exceed_prob(self, order):

        match order:
            case 0:
                Ck = 1 - norm.cdf( (self.lnGMi - self.lnmed) / np.sqrt(self.sigma_mu**2 + self.lnsigma**2) )
                return Ck
            case 1:
                const = self.alpha
                numer = 1
                denom = (-self.a)**(1/2)
            case 2:
                const = (self.alpha / 2) 
                numer = self.b
                denom = 2*(-self.a)**(3/2)
            case 3:
                const = (self.alpha / 6) 
                numer = -2*self.a*(1+2*self.a) + self.b**2
                denom = 4 * (-self.a)**(5/2)
            case 4:
                const = (self.alpha / 24) 
                numer = -self.b * ( 6*self.a*(1+2*self.a) - self.b**2 )
                denom = 8 * (-self.a)**(7/2)
            case 5:
                const = (self.alpha / 120) 
                numer = 12 * self.a**2 * (1+2*self.a)**2 -12*self.a*(1+2*self.a)*self.b**2 + self.b**4
                denom = 16 * (-self.a)**(9/2)
            case 6:
                const = (self.alpha / 720) 
                numer = self.b * 60*self.a**2 * (1+2*self.a)**2 - 20*self.a*(1+2*self.a)*self.b**2 + self.b**4
                denom = 32 * (-self.a)**(11/2)
            case _:
                raise ValueError("order should be 1, 2, ..., 6")
            
        
        
        Ck = const * numer / denom 

        return Ck
    

    def run_PCE(self):
        """
        x = standard normal random variable
        """
        y_hat = np.zeros((self.Nmarginal, self.Nxi))
        x = np.sort(np.random.normal(size=self.Nxi))
        
        for order in range(0,self.max_order+1):
            c = self.coeff_exceed_prob(order)
            
            psi = self.hermite_poly(x, order = order)
            
            y_hat += c[:,np.newaxis] * psi
            #print("yhat", y_hat_order)
            #y_hat+=c*psi
            
        return y_hat


def run_haz_PCEmedian(X, site = [0,0,0], GMi=1.0, sigma_mu=0.2, max_order = 6, rate=1.0, outputs=['mean', 10, 50, 90]):
    m, x, y, z = X.T
    pdf = np.full_like(m, 0.1)
    r = ( (site[0] - x)**2 + (site[1]-y)**2 + (site[2]-z)**2 )**0.5
    lnmed, lnsigma = GroundMotionModel(model='sadigh1997').calc_mean_sigma(m, r)
    #lnsigma = np.full_like(m, 0.5)
    
    pcemodel = PCE_medianGMM(GMi=GMi, lnmed=lnmed, lnsigma=lnsigma, sigma_mu=sigma_mu, max_order = max_order) ### lnmed and lnsigma can be vectors
    P = pcemodel.run_PCE()
    
    y_hat = rate*P*pdf[:, np.newaxis]
    haz = np.sum(y_hat, axis=0)

    outhaz = []
    for output in outputs:
        match output:
            case 'mean':
                outhaz.append( np.mean(haz) )
            case int() as num if 1 <= num <= 99:
                outhaz.append( np.percentile(haz, output) )
            case '_':
                raise ValueError("Invalid output")

    return haz, outhaz

#X = np.array([[5.5, 10, 10, 10],[6, 20, 20, 5]])
#X = np.array([[5.5, 10, 10, 10]])
#haz, outhaz = run_haz_PCEmedian(X)
#print("haz", haz)
#print("outhaz", outhaz)


'''def run_haz_PCEmedian(X, site = [0,0,0], GMi=0.1, sigma_mu=0.2, Nxi = 1000, max_order = 6, rate=1.0, outputs=['mean', 10, 50, 90]):
    xi = np.random.normal(loc=0, scale=1, size=Nxi)
    y_hat = np.zeros( (len(X), Nxi) )

    for i, EQscenario in enumerate(X):
        m, x, y, z = EQscenario.T
        pdf = 0.1
        r = ( (site[0] - x)**2 + (site[1]-y)**2 + (site[2]-z)**2 )**0.5
        lnmed, lnsigma = calc_mean_sigma_sadigh1997(m, r)
        pcemodel = PCE_medianGMM(GMi=GMi, lnmed=lnmed, lnsigma=lnsigma, sigma_mu=sigma_mu, max_order = max_order)
        P = pcemodel.run_PCE(xi)
        y_hat[i,:] = rate*P*pdf
    haz = np.sum(y_hat, axis=0)

    outhaz = []
    for output in outputs:
        match output:
            case 'mean':
                outhaz.append( np.mean(haz) )
            case int() as num if 1 <= num <= 99:
                outhaz.append( np.percentile(haz, output) )
            case '_':
                raise ValueError("Invalid output")

    return haz, outhaz


def run_haz_PCEmedian(X, site = [0,0,0], GMi=0.1, sigma_mu=0.2, Nxi = 1000, max_order = 6, rate=1.0, outputs=['mean', 10, 50, 90]):
    xi = np.random.normal(loc=0, scale=1, size=Nxi)
    y_hat = np.zeros( (len(X), Nxi) )

    for i, EQscenario in enumerate(X):
        m, x, y, z = EQscenario.T
        pdf = 0.1
        r = ( (site[0] - x)**2 + (site[1]-y)**2 + (site[2]-z)**2 )**0.5
        lnmed, lnsigma = calc_mean_sigma_sadigh1997(m, r)
        pcemodel = PCE_medianGMM(GMi=GMi, lnmed=lnmed, lnsigma=lnsigma, sigma_mu=sigma_mu, max_order = max_order)
        P = pcemodel.run_PCE(xi)
        y_hat[i,:] = rate*P*pdf
    haz = np.sum(y_hat, axis=0)

    outhaz = []
    for output in outputs:
        match output:
            case 'mean':
                outhaz.append( np.mean(haz) )
            case int() as num if 1 <= num <= 99:
                outhaz.append( np.percentile(haz, output) )
            case '_':
                raise ValueError("Invalid output")

    return haz, outhaz


def run_haz_PCEmedian_for_AIS(X, site = [0,0,0], GMi=0.1, sigma_mu=0.2, Nxi = 1000, max_order = 6, rate=1.0, output='mean'):
    xi = np.random.normal(loc=0, scale=1, size=Nxi)
    y_hat = np.zeros( (len(X), Nxi) )

    for i, EQscenario in enumerate(X):
        m, x, y, z = EQscenario.T
        pdf = 0.1
        r = ( (site[0] - x)**2 + (site[1]-y)**2 + (site[2]-z)**2 )**0.5
        lnmed, lnsigma = calc_mean_sigma_sadigh1997(m, r)
        pcemodel = PCE_medianGMM(GMi=GMi, lnmed=lnmed, lnsigma=lnsigma, sigma_mu=sigma_mu, max_order = max_order)
        P = pcemodel.run_PCE(xi)
        y_hat[i,:] = rate*P*pdf
    haz = np.sum(y_hat, axis=0)

    
    match output:
        case 'mean':
            outhaz = np.mean(haz)
        case int() as num if 1 <= num <= 99:
            outhaz = np.percentile(haz, output)
        case '_':
            raise ValueError("Invalid output")

    return outhaz

def run_haz_MCmedian(X, site = [0,0,0], GMi=0.1, sigma_mu=0.2, N = 1000, rate=1):
    dmus = np.random.normal(loc=0, scale=sigma_mu, size=N)
    y_margin = np.zeros( (len(X), N) )
    
    for j, dmu in enumerate(dmus):
        for i, EQscenario in enumerate(X):
            m, x, y, z = EQscenario.T
            pdf = 0.1
            r = ( (site[0] - x)**2 + (site[1]-y)**2 + (site[2]-z)**2 )**0.5
            lnmed, lnsigma = calc_mean_sigma_sadigh1997(m, r)
            lnmed+=dmu
            lnGM = np.log(GMi)
            P = 1 - norm.cdf( (lnGM - lnmed)/lnsigma )
            y_margin[i,j] += rate*P*pdf
    haz = np.sum(y_margin, axis=0)

    return haz
        
EQscenarios = np.array([[6.0, 10, 10, 5], [7.0, 20, 20, 5]]) ### rate, magnitude, location
GMi_list = np.power(10, np.arange(-1, 1.1, 0.1))
sigma_mu = 0.2

########## PCE ##################
haz_10th_PCE_list = []
haz_50th_PCE_list = []
haz_90th_PCE_list = []
haz_mean_PCE_list = []

Nxi = 100
for GMi in GMi_list:
    outhaz = run_haz_PCEmedian_for_AIS(EQscenarios, GMi = GMi, Nxi = Nxi, sigma_mu = sigma_mu, max_order=6,outputs=['mean', 10, 50, 90])
    haz_mean_PCE_list.append(outhaz[0])
    haz_10th_PCE_list.append(outhaz[1])
    haz_50th_PCE_list.append(outhaz[2])
    haz_90th_PCE_list.append(outhaz[3])
    
###############################
#print(haz_10th_PCE_list)


### Plotting ####
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(GMi_list, haz_mean_MC_list, color='k', label = "MC")
ax.plot(GMi_list, haz_10th_MC_list, color='gray')
ax.plot(GMi_list, haz_50th_MC_list, color='gray')
ax.plot(GMi_list, haz_90th_MC_list, color='gray')

ax.scatter(GMi_list, haz_mean_PCE_list, color='k', marker='.')
ax.scatter(GMi_list, haz_10th_PCE_list, color='gray', marker='s')
ax.scatter(GMi_list, haz_50th_PCE_list, color='gray', marker='^')
ax.scatter(GMi_list, haz_90th_PCE_list, color='gray', marker='o')

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-6, 1)
plt.show()

exit()'''


