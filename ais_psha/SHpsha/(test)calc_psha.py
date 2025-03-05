from .model_gmm import *
#from .model_magnitude import *
from .model_sourcelocation import *
#from .model_sourceprob import *

import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from collections import Counter

class PSHA:
    def __init__(self, sources, site = [0,0,0], GMi_list=[0.001, 0.01, 0.1, 1]):
        self.site=site
        self.sources = sources
        self.Nsources = len(self.sources)
        self.sourceprob = RateModel(self.sources)
        self.rate_tot = self.sourceprob.rate_tot
        self.GMi_list = GMi_list

    def sort_src_sample(self, x):
        x = x.astype(int)

        mask = (x<1) | (x>self.sourceprob.Nsrc)
        x[mask] = 0

        counts = np.bincount(x)
        id = np.unique(x)
        Nsmpl_list_by_src = counts[counts>0]
        Nsmpl_cumsum_list_by_src = np.cumsum(Nsmpl_list_by_src)

        return id, Nsmpl_list_by_src, Nsmpl_cumsum_list_by_src

    
    def sample(self, Nsmpl=100):
        Ndim = 6 ####### src, m, x, y, z, eps
        
        ## Sample and Source and Sort ###
        src = self.sourceprob.sample(Nsmpl=Nsmpl)
        id_src_sorted, Nsmpl_src_sorted, Nsmpl_cumsum_src_sorted = self.sort_src_sample(src)
        #######################

        ### Sample m, loc, eps for each source ###
        samples = np.zeros((Nsmpl,Ndim))
        
        i=0
        for id_src, Nsmpl_src in np.c_[id_src_sorted, Nsmpl_src_sorted]:
            m = self.sources[id_src]["m"].sample(Nsmpl_src)
            xyz = self.sources[id_src]["loc"].sample(Nsmpl_src)
            eps = self.sources[id_src]["gmm"].sample(Nsmpl=Nsmpl_src)

            
            ## Append the samples ##
            start_idx = 0 if i == 0 else Nsmpl_cumsum_src_sorted[i - 1]
            end_idx = Nsmpl_cumsum_src_sorted[i]
            samples[start_idx:end_idx, 0] = id_src
            samples[start_idx:end_idx, 1] = m
            samples[start_idx:end_idx, 2:5] = xyz  # Assign xyz[:,0], xyz[:,1], xyz[:,2] at once
            samples[start_idx:end_idx, 5] = eps
            i+=1

        samples = np.array(samples)
        
        return samples
    
    def simulate_gm(self, X):
        src, m, x, y, z, eps = X.T
        Ndim = 6 ####### src, m, x, y, z, eps
        Nsmpl = len(X)

        ## Sort src ###
        id_src_sorted, Nsmpl_src_sorted, Nsmpl_cumsum_src_sorted = self.sort_src_sample(src)
        #######################

        ### Sample m, loc, eps for each source ###
        gms = np.zeros(Nsmpl)
        
        i=0
        for id_src in id_src_sorted:
            
            start_idx = 0 if i == 0 else Nsmpl_cumsum_src_sorted[i - 1]
            end_idx = Nsmpl_cumsum_src_sorted[i]
            m_sel=m[start_idx:end_idx]
            x_sel=x[start_idx:end_idx]
            y_sel=y[start_idx:end_idx]
            z_sel=z[start_idx:end_idx]
            eps_sel = eps[start_idx:end_idx]
            
            r_sel = np.sqrt( (x_sel-self.site[0])**2 + (y_sel-self.site[1])**2 + (z_sel-self.site[2])**2 ) ### This could be the function in the future
            
            if 0<id_src<self.Nsources+1:
                gm = self.sources[id_src]["gmm"].simulate(m_sel, r_sel, eps_sel)
            else:
                gm = np.zeros(end_idx-start_idx)
            
            ## Append the samples ##
            gms[start_idx:end_idx] = gm
            i+=1

        gms = np.array(gms)    
        return gms
    

    def haz(self, Nhazsmpl=100, method="MC",
            Ngaussian=1, Niter=5, integ_range=[[]] ### Params for PMC
            ):

        Pf = np.array([])
        cov = np.array([])
        match method:
            case 'MC':
                x = self.sample(Nsmpl=Nhazsmpl)
                gm = self.simulate_gm(x)
                for GMi in self.GMi_list:
                    integrands = (gm-GMi) > 0
                    Nexceed = np.sum( integrands )
                    Pf =  np.append(Pf, Nexceed / Nhazsmpl)
                    
                    cov = np.append( cov, np.sqrt( (1 - Pf)/(Nhazsmpl*Pf) ) )
            case 'PMC':
                from SHpmc import Run_pmc_normal_integ3
                for GMi in self.GMi_list:
                    out=Run_pmc_normal_integ3(self.integrand, integ_range, Nsmpl=Nhazsmpl, Niter=Niter, GMi=GMi)
                    cov = np.append(cov, out.COV)
                    Pf = np.append(Pf, out.integ)
                    
        hazard = Pf*self.rate_tot
        return hazard, cov
    
    def integrand(self, x, GMi=0.1):        
        fX = self.pdf(x)
        gm = self.simulate_gm(x)
        I = np.where(gm-GMi>0, 1, 0)

        return I*fX

    def pdf(self, X):
        src, m, x, y, z, eps = X.T
        NX = len(src)

        ## Sort Source ###
        id_src_sorted, Nsmpl_src_sorted, Nsmpl_cumsum_src_sorted = self.sort_src_sample(src)
        #######################

        ### Sample m, loc, eps for each source ###
        i=0
        f_list = np.zeros(NX)
        
        for id_src, Nsmpl_src in np.c_[id_src_sorted, Nsmpl_src_sorted]:
            
            start_idx = 0 if i == 0 else Nsmpl_cumsum_src_sorted[i - 1]
            end_idx = Nsmpl_cumsum_src_sorted[i]

            if 0 < id_src < self.Nsources+1:
                f_src = self.sourceprob.pdf(id_src)
                f_m = self.sources[id_src]["m"].pdf(m[start_idx:end_idx])
                f_xyz = self.sources[id_src]["loc"].pdf(np.c_[x,y,z][start_idx:end_idx])
                f_eps = self.sources[id_src]["gmm"].pdf(eps[start_idx:end_idx])
            else:
                f_src = 0
                f_m = np.zeros(end_idx - start_idx)
                f_xyz = np.zeros(end_idx - start_idx)
                f_eps = np.zeros(end_idx - start_idx)
            f = f_src * f_m * f_xyz * f_eps

            f_list[start_idx:end_idx] = f
            i+=1
            #plt.scatter(id_src,f_src)
        #plt.show()
        
        #f_list *= f_src
        

        return f_list
