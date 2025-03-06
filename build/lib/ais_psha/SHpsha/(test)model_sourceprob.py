from .model_gmm import *
#from .model_magnitude import *
from .model_sourcelocation import *

import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

class RateModel:
    def __init__(self, sources):
        
        rate_tot = 0.
        for src in sources.keys():
            sources[src]["rate"] = sources[src]["m"].calc_rate()
            rate_tot+=sources[src]["rate"]
        self.rate_tot = rate_tot

        
        src_prob=[]
        for src in sources.keys():
            sources[src]["prob"] = sources[src]["rate"]/rate_tot
            sources[src]["pdf"] = sources[src]["rate"]/rate_tot
            src_prob.append(sources[src]["prob"])
        src_prob = np.array(src_prob)
        self.src_prob = src_prob
        self.src_pdf = src_prob
        
        self.sources = sources
        
        #self.src_cdf = np.cumsum(src_prob)
        self.Nsrc = len(self.sources)
        self.integ_range = [0,self.Nsrc]
        

    def sample(self, Nsmpl=100):
        edges = np.arange(0,self.Nsrc+1,1)

        widths = np.diff(edges)  # Bin widths
        cdf = np.cumsum(self.src_prob)     # Compute cumulative distribution function (CDF)

        # Generate uniform samples and find corresponding bins
        u = np.random.rand(Nsmpl)
        indices = np.searchsorted(cdf, u)

        # Sample uniformly within the chosen bin
        samples = edges[indices] + np.random.rand(Nsmpl) * widths[indices]
        samples = np.random.choice(range(1, self.Nsrc+1), size=Nsmpl, p=self.src_prob)
        samples = samples.astype(int)
        samples = np.sort(samples)

        return samples
      
    def pdf(self, x):
        #print(x)
        #x = np.array(x)
        #x = x.astype(int)+1
        #idx = x-1
        #mask = (idx>=0) & (idx<self.Nsrc)
        #print(idx_mask)

        #f=np.zeros_like(idx, dtype=float)
        #f[mask] = self.src_pdf[idx[mask]]
        #print("uniq", np.unique(x))
        #print(self.src_pdf)
        #f = np.where(x<=self.Nsrc, self.src_pdf[idx], 0)
        f = self.sources[x]["pdf"]
        #f = self.src_pdf[x-1]
        #print("f", f)
        print(f)
        return f
        