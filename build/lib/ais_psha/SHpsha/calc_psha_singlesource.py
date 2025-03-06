from .model_gmm import *
#from .model_magnitude import *
from .model_sourcelocation import *
#from .model_sourceprob import *

import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from collections import Counter

from ais_psha.SHvegas import Run_vegas_integ
from ais_psha.SHpmc import Run_pmc_normal_integ3

class PSHA_singlesource:
    def __init__(self, sources, site = [0,0,0], GMi_list=[0.001, 0.01, 0.1, 1]):
        self.site=site
        self.sources = sources
        self.Nsources = len(self.sources)
        self.rate_tot = self.sources[1]["rate"]

        self.GMi_list = GMi_list
        
        self.sample_eps = self.sources[1]["gmm"].sample_eps
        

        if self.sample_eps == True:
            self.sample = self.sample
            self.pdf = self.pdf
            self.haz = self.haz
            self.simulate_gm = self.simulate_gm
            self.integrand = self.integrand
        else:
            self.sample = self.sample_no_eps
            self.pdf = self.pdf_no_eps
            self.haz = self.haz_no_eps
            self.simulate_gm = self.simulate_gm_no_eps
            self.integrand = self.integrand_no_eps

    
    def sample(self, Nsmpl=100):
        Ndim = 5 ####### m, x, y, z, eps

        ### Sample m, loc, eps for each source ###
        samples = np.zeros((Nsmpl,Ndim))
        
        id_src=1
        m = self.sources[id_src]["m"].sample(Nsmpl)
        xyz = self.sources[id_src]["loc"].sample(Nsmpl)
        eps = self.sources[id_src]["gmm"].sample(Nsmpl=Nsmpl)

        samples = np.c_[m, xyz[:,0], xyz[:,1], xyz[:,2], eps]
        
        return samples
    
    def simulate_gm(self, X):
        m, x, y, z, eps = X.T
        Nsmpl = len(X)
##

        ### Sample m, loc, eps for each source ###
        gms = np.zeros(Nsmpl)
        
        id_src = 1          
        r_sel = np.sqrt( (x-self.site[0])**2 + (y-self.site[1])**2 + (z-self.site[2])**2 ) ### This could be the function in the future
            
        gms = self.sources[id_src]["gmm"].simulate(m, r_sel, eps)    
        gms = np.array(gms)

        return gms
    

    def haz(self, Nhazsmpl=100, method="MC",
            Ngaussian=1, Niter=5, integ_range=[[]] ### Params for PMC
            ):

        Pf_list = np.array([])
        cov_list = np.array([])

        match method:
            case 'MC':
                x = self.sample(Nsmpl=Nhazsmpl)
                gm = self.simulate_gm(x)
                for GMi in self.GMi_list:
                    integrands = (gm-GMi) > 0
                    Nexceed = np.sum( integrands )
                    Pf = Nexceed / Nhazsmpl
                    cov = np.sqrt( (1 - Pf)/(Nhazsmpl*Pf) )
                    Pf_list =  np.append(Pf_list, Pf)
                    cov_list = np.append(cov_list, cov)
                    
            case 'PMC':
                for GMi in self.GMi_list:
                    out=Run_pmc_normal_integ3(self.integrand, integ_range, Nsmpl=Nhazsmpl, Niter=Niter, GMi=GMi, show_integ=False)
                    Pf = out.integ
                    cov = out.COV
                    Pf_list =  np.append(Pf_list, Pf)
                    cov_list = np.append(cov_list, cov)
            
            case 'vegas':
                Ngrid = np.full(len(integ_range), 50)
                for GMi in self.GMi_list:
                    out=Run_vegas_integ(self.integrand, integ_range, Ngrid, Nhazsmpl, Niter=10, print_output=False)
                    
                    Pf = out.integration
                    cov = out.COV
                    Pf_list =  np.append(Pf_list, Pf)
                    cov_list = np.append(cov_list, cov)
        
        
        hazard_list = Pf_list*self.rate_tot
        return hazard_list, cov_list
    
    def integrand(self, x, GMi=0.001):
        fX = self.pdf(x)
        gm = self.simulate_gm(x)
        I = np.where(gm-GMi>0, 1, 0)

        return I*fX

    def pdf(self, X):
        m, x, y, z, eps = X.T
        
        id_src = 1
        
        f_m = self.sources[id_src]["m"].pdf(m)
        f_xyz = self.sources[id_src]["loc"].pdf(np.c_[x,y,z])
        f_eps = self.sources[id_src]["gmm"].pdf(eps)
        
        f = f_m * f_xyz * f_eps

        return f
    


    def sample_no_eps(self, Nsmpl=100):
        Ndim = 4 ####### m, x, y, z, eps

        ### Sample m, loc, eps for each source ###
        samples = np.zeros((Nsmpl,Ndim))
        
        id_src=1
        m = self.sources[id_src]["m"].sample(Nsmpl)
        xyz = self.sources[id_src]["loc"].sample(Nsmpl)
        #eps = self.sources[id_src]["gmm"].sample(Nsmpl=Nsmpl)

        samples = np.c_[m, xyz[:,0], xyz[:,1], xyz[:,2]]
        
        return samples
    
    def simulate_gm_no_eps(self, X):
        m, x, y, z = X.T
        
        id_src = 1          
        r_sel = np.sqrt( (x-self.site[0])**2 + (y-self.site[1])**2 + (z-self.site[2])**2 ) ### This could be the function in the future
            
        lnmed, lnsigma = self.sources[id_src]["gmm"].simulate(m, r_sel)    
        
        return lnmed, lnsigma #prob_ex_list
        

    def haz_no_eps(self, Nhazsmpl=100, method="MC",
            Ngaussian=1, Niter=5, integ_range=[[]] ### Params for PMC
            ):

        Pf_list = np.array([])
        cov_list = np.array([])
        match method:
            case 'MC':
                x = self.sample(Nsmpl=Nhazsmpl)
                lnmed, lnsigma = self.simulate_gm(x)
                for GMi in self.GMi_list:
                    z = (np.log(GMi) - lnmed) / lnsigma
                    integrands = 1 - norm.cdf(z)
                    Pf = np.mean(integrands)

                    # Assuming integrands, Nhazsmpl, and Pf are defined
                    bootstrap_samples = np.random.choice(integrands, size=(200, Nhazsmpl), replace=True)
                    bootstrap_estimates = np.mean(bootstrap_samples, axis=1)
                    cov = np.std(bootstrap_estimates) / Pf


                    #sigma = np.std(integrands)
                    
                    #cov = np.sqrt( (1 - Pf)/(Nhazsmpl*Pf) )
                    #cov = sigma / Pf
                    Pf_list =  np.append(Pf_list, Pf)
                    cov_list = np.append(cov_list, cov)
                    
            case 'PMC':
                from ais_psha.SHpmc import Run_pmc_normal_integ3
                for GMi in self.GMi_list:
                    out=Run_pmc_normal_integ3(self.integrand, integ_range, Nsmpl=Nhazsmpl, Niter=Niter, GMi=GMi, show_integ=False)
                    Pf = out.integ
                    cov = out.COV
                    Pf_list =  np.append(Pf_list, Pf)
                    cov_list = np.append(cov_list, cov)

            case 'vegas':
                from ais_psha.SHvegas import Run_vegas_integ
                #print("vegas")
                Ngrid = np.full(len(integ_range), 50)
                for GMi in self.GMi_list:
                    out=Run_vegas_integ(self.integrand, integ_range, Ngrid, Nhazsmpl, Niter=10, GMi=GMi, print_output=False)
                    #print(out)
                    Pf = out.integration[-1]
                    cov = out.COV[-1]
                    Pf_list =  np.append(Pf_list, Pf)
                    cov_list = np.append(cov_list, cov)
        
        
        hazard_list = Pf_list*self.rate_tot
        return hazard_list, cov_list
    
    def integrand_no_eps(self, x, GMi=0.001):
        fX = self.pdf(x)
        lnmed, lnsigma = self.simulate_gm(x)
        z = (np.log(GMi) - lnmed) / lnsigma
        P = 1 - norm.cdf(z)

        return P*fX

    def pdf_no_eps(self, X):
        m, x, y, z = X.T
        
        id_src = 1
        
        f_m = self.sources[id_src]["m"].pdf(m)
        f_xyz = self.sources[id_src]["loc"].pdf(np.c_[x,y,z])
        
        f = f_m * f_xyz 

        return f

        
