from .model_gmm import *
#from .model_magnitude import *
from .model_sourcelocation import *
#from .model_sourceprob import *
from .pce import *

#import matplotlib.pyplot as plt
#import scipy.stats as stats
#from sklearn.mixture import GaussianMixture
#from collections import Counter

from SHvegas import Run_vegas_integ
from SHpmc import Run_pmc_normal_integ3

class PSHA_singlesource_medianPCE:
    def __init__(self, sources, site = [0,0,0], GMi_list=[0.001, 0.01, 0.1, 1], sigma_mu = 0.2, Nxi = 1000):
        self.site=site
        self.sources = sources
        self.Nsources = len(self.sources)
        self.rate_tot = self.sources[1]["rate"]
        self.GMi_list = GMi_list
        self.sample_eps = self.sources[1]["gmm"].sample_eps
        self.Nxi = Nxi
        

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

        self.sigma_mu = sigma_mu

    
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
    
    def simulate_gm(self, X):
        m, x, y, z, eps = X.T
        Nsmpl = len(X)

        ### Sample m, loc, eps for each source ###
        gms = np.zeros(Nsmpl)
        
        id_src = 1          
        r_sel = np.sqrt( (x-self.site[0])**2 + (y-self.site[1])**2 + (z-self.site[2])**2 ) ### This could be the function in the future
            
        gms = self.sources[id_src]["gmm"].simulate(m, r_sel, eps)    
        gms = np.array(gms)

        return gms
    
    def simulate_gm_no_eps(self, X):
        m, x, y, z = X.T
        
        id_src = 1          
        r_sel = np.sqrt( (x-self.site[0])**2 + (y-self.site[1])**2 + (z-self.site[2])**2 ) ### This could be the function in the future
            
        lnmed, lnsigma = self.sources[id_src]["gmm"].simulate(m, r_sel)    
        
        return lnmed, lnsigma #prob_ex_list
    

    def haz(self, Nhazsmpl=100, method="MC",
            Ngaussian=1, Niter=5, integ_range=[[]] ### Params for PMC
            ):

        Pf_list = np.array([])
        cov_list = np.array([])
        N_list = np.array([])
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
                    N_list = np.append(N_list, Nhazsmpl)
                    
            case 'PMC':
                from SHpmc import Run_pmc_normal_integ3
                for GMi in self.GMi_list:
                    out=Run_pmc_normal_integ3(self.integrand, integ_range, Nsmpl=Nhazsmpl, Niter=Niter, GMi=GMi, show_integ=True)
                    Pf = out.integ
                    cov = out.COV
                    Pf_list =  np.append(Pf_list, Pf)
                    cov_list = np.append(cov_list, cov)
                    N_list = np.append(N_list, Nhazsmpl*out.Niter)
        
        
        hazard_list = Pf_list*self.rate_tot
        return hazard_list, cov_list, N_list
    

    def haz_no_eps(self, Nhazsmpl=100, method="MC",
            Ngaussian=1, Niter=5, integ_range=[[]] ### Params for PMC
            ):

        Pf_list = np.array([])
        cov_list = np.array([])
        N_list = np.array([])
        match method:
            case 'MC':
                x = self.sample(Nsmpl=Nhazsmpl)
                lnmed, lnsigma = self.simulate_gm(x)
                for GMi in self.GMi_list:
                    z = (np.log(GMi) - lnmed) / lnsigma
                    integrands = 1 - norm.cdf(z)
                    
                    Pf = np.mean(integrands)
                    cov = np.sqrt( (1 - Pf)/(Nhazsmpl*Pf) )
                    Pf_list =  np.append(Pf_list, Pf)
                    cov_list = np.append(cov_list, cov)
                    N_list = np.append(N_list, Nhazsmpl)
                    
            case 'PMC':
                from SHpmc import Run_pmc_normal_integ3
                for GMi in self.GMi_list:
                    out=Run_pmc_normal_integ3(self.integrand, integ_range, Nsmpl=Nhazsmpl, Niter=Niter, GMi=GMi, show_integ=True)
                    Pf = out.integ
                    cov = out.COV
                    Pf_list =  np.append(Pf_list, Pf)
                    cov_list = np.append(cov_list, cov)
                    N_list = np.append(N_list, Nhazsmpl*out.Niter)
        
        
        hazard_list = Pf_list*self.rate_tot
        return hazard_list, cov_list, N_list
    
    def integrand(self, x, GMi=0.001):
        fX = self.pdf(x)
        gm = self.simulate_gm(x)
        I = np.where(gm-GMi>0, 1, 0)

        return I*fX
    
    def integrand_no_eps(self, x, GMi=0.001):
        fX = self.pdf(x)
        lnmed, lnsigma = self.simulate_gm(x)
        z = (np.log(GMi) - lnmed) / lnsigma
        P = 1 - norm.cdf(z)

        return P*fX

    def pdf(self, X):
        m, x, y, z, eps = X.T
        
        id_src = 1
        
        f_m = self.sources[id_src]["m"].pdf(m)
        f_xyz = self.sources[id_src]["loc"].pdf(np.c_[x,y,z])
        f_eps = self.sources[id_src]["gmm"].pdf(eps)
        
        f = f_m * f_xyz * f_eps

        return f

    def pdf_no_eps(self, X):
        m, x, y, z = X.T
        
        id_src = 1
        
        f_m = self.sources[id_src]["m"].pdf(m)
        f_xyz = self.sources[id_src]["loc"].pdf(np.c_[x,y,z])
        
        f = f_m * f_xyz 

        return f
    

    def haz_PCEmedian(self, integ_range=[[]], Ngrid = [], Nhazsmpl = 100, output='mean', method = 'MC'):
        
        haz_list = []
        cov_list = []
        Ntot_list = []
        X = self.sample(Nsmpl = Nhazsmpl)
        for GMi in self.GMi_list:
            if method == 'vegas':
                
                out = Run_vegas_integ(self.integrand_PCEmedian_w_pdf, integ_range, Ngrid , Nsmpl = Nhazsmpl, print_output= False,
                                      sigma_mu = self.sigma_mu, GMi=GMi, output=output)
                y = out.integration[-1]
                cov = out.COV[-1]
                Ntot =  Nhazsmpl # out.Kopt *
                haz = self.rate_tot*y
            elif method == 'PMC':
                out = Run_pmc_normal_integ3(self.integrand_PCEmedian_w_pdf, integ_range=integ_range, Nsmpl = Nhazsmpl, show_integ=True,
                                            sigma_mu = self.sigma_mu, GMi=GMi, output=output)
                y = out.integ
                cov = out.COV
                Ntot =  Nhazsmpl # out.Niter *
                haz = self.rate_tot*y
            elif method == 'MC':
                out = self.integrand_PCEmedian_wo_pdf(X, sigma_mu = self.sigma_mu, GMi=GMi, output=output)
                y = np.mean(out)
                cov = np.sqrt( (1 - y) / (Nhazsmpl*y) )
                Ntot = Nhazsmpl
                haz = self.rate_tot*y

            haz_list.append(haz)
            cov_list.append(cov)
            Ntot_list.append(Ntot)
            
        return haz_list, cov_list, Ntot_list

    def integrand_PCEmedian_wo_pdf(self, X, sigma_mu=0.2, GMi=0.1,  max_order = 6, output='mean'):
        m, x, y, z = X.T
        pdf = self.pdf(X)
        r = ( (self.site[0] - x)**2 + (self.site[1]-y)**2 + (self.site[2]-z)**2 )**0.5
        lnmed, lnsigma = self.simulate_gm(X)
        #outhaz = []
        
        pcemodel = PCE_medianGMM(GMi=GMi, lnmed=lnmed, lnsigma=lnsigma, sigma_mu=sigma_mu, max_order = max_order, Nxi=self.Nxi)
        P = pcemodel.run_PCE()
    
        Pf_hat = P

        match output:
            case 'mean':
                outhaz = np.mean(Pf_hat, axis=1)
            case int() as num if 1 <= num <= 99:
                outhaz = np.percentile(Pf_hat, num, axis=1)
            case '_':
                raise ValueError("Invalid output")
        return outhaz

    def integrand_PCEmedian_w_pdf(self, X, sigma_mu=0.2, GMi=0.1,  max_order = 6, output='mean'):
        m, x, y, z = X.T
        pdf = self.pdf(X)
        r = ( (self.site[0] - x)**2 + (self.site[1]-y)**2 + (self.site[2]-z)**2 )**0.5
        lnmed, lnsigma = self.simulate_gm(X)
        
        pcemodel = PCE_medianGMM(GMi=GMi, lnmed=lnmed, lnsigma=lnsigma, sigma_mu=sigma_mu, max_order = max_order, Nxi=self.Nxi)
        P = pcemodel.run_PCE()
    
        Pf_hat = P * pdf[:, np.newaxis]

        match output:
            case 'mean':
                outhaz = np.mean(Pf_hat, axis=1)
            case int() as num if 1 <= num <= 99:
                outhaz = np.percentile(Pf_hat, num, axis=1)
            case '_':
                raise ValueError("Invalid output")
        return outhaz