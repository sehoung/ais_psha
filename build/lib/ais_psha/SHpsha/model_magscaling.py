import numpy as np
class MagScaling():
    def __init__(self, model):
        self.model = model.lower()
        
    def M2A(self, x):
        eps = np.random.normal(loc=0, scale=1, size=len(x))
        match self.model:
            case "PEERTEST":
                return np.power(10, x-4) + 0*eps
    
    def M2W(self, x):
        eps = np.random.normal(loc=0, scale=1, size=len(x))
        match self.model:
            case "PEERTEST":
                return np.power(0.5*x - 2.15) + 0*eps
            
    def M2L(self, x):
        eps = np.random.normal(loc=0, scale=1, size=len(x))
        match self.model:
            case "PEERTEST":
                return np.power(0.5*x - 1.85) + 0*eps
            
    def A2M(self, x):
        eps = np.random.normal(loc=0, scale=1, size=len(x))
        match self.model:
            case "PEERTEST":
                return np.log10(x) + 4 + 0*eps
    
    def W2M(self, x):
        eps = np.random.normal(loc=0, scale=1, size=len(x))
        match self.model:
            case "PEERTEST":
                return (np.log10(x) + 2.15)/0.5 + 0*eps
            
    def L2M(self, x):
        eps = np.random.normal(loc=0, scale=1, size=len(x))
        match self.model:
            case "PEERTEST":
                return (np.log10(x) + 1.85)/0.5 + 0*eps
     
            
    
            