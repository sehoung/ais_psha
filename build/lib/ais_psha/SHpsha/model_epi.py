import numpy as np
import scipy.stats as stats

class PSHAepi:
    def __init__(self, dist_type, **kwargs):
        """
        Initialize the distribution.
        
        Parameters:
        - dist_type: 'normal', 'truncnorm', or 'uniform'
        - kwargs: Parameters required for each distribution.
            - Normal: mean (loc), std (scale)
            - Truncated Normal: mean (loc), std (scale), lower (a), upper (b)
            - Uniform: lower (low), upper (high)
        """
        self.dist_type = dist_type.lower()
        
        if self.dist_type == "normal":
            self.mean = kwargs.get("loc", 0)
            self.std = kwargs.get("scale", 1)
            self.dist = stats.norm(self.mean, self.std)
        
        elif self.dist_type == "truncnorm":
            self.mean = kwargs.get("loc", 0)
            self.std = kwargs.get("scale", 1)
            self.lower = kwargs.get("a", -np.inf)
            self.upper = kwargs.get("b", np.inf)
            self.a, self.b = (self.lower - self.mean) / self.std, (self.upper - self.mean) / self.std
            self.dist = stats.truncnorm(self.a, self.b, loc=self.mean, scale=self.std)
        
        elif self.dist_type == "uniform":
            self.low = kwargs.get("low", 0)
            self.high = kwargs.get("high", 1)
            self.dist = stats.uniform(self.low, self.high - self.low)
        
        else:
            raise ValueError("Unsupported distribution type. Choose from 'normal', 'truncnorm', or 'uniform'.")

    def pdf(self, x):
        """Returns the probability density function (PDF) at x."""
        return self.dist.pdf(x)

    def sample(self, n=1):
        """Generates n random samples from the distribution."""
        return self.dist.rvs(size=n)
