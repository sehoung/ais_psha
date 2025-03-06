import numpy as np
from scipy.stats import truncnorm

class MagProb():
    def __init__(self, model='exp', mmin = None, mmax=None, b=None, mchar_min=None, mchar_max = None, dm_equi = None, mchar_avg=None, mchar_stdev = None):
        """
        model 'exp': (b, mmin, self.mmax)
        model 'YC': (b, mchar_min, mchar_max, dm_equi), n(Mchar) = n(Mchar_min - dm_equi)
        model 'PureChar': No small earthquake (mchar_avg, mchar_stdev, mchar_min, mchar_max)
        """
        self.model = model.lower()
        
        match self.model:
            case 'exp':
                if mmin is None or mmax is None or b is None:
                    raise ValueError("For model 'exp', mmin, mmax, and b should be provided")
                self.mmin = mmin
                self.mmax = mmax
                self.b = b
            case 'yc':
                if mmin is None or mchar_min is None or mchar_max is None or dm_equi is None or b is None:
                    raise ValueError("For model 'exp', mmin, mchar_min, mchar_mmax, and b should be provided")
                self.mmin = mmin
                self.mchar_min = mchar_min
                self.mchar_max = mchar_max
                self.dm_equi = dm_equi
                self.b = b
            case 'purechar':
                if mmin is None or mchar_avg is None or mchar_stdev is None or mchar_min is None or mchar_max is None:
                    raise ValueError("For model 'exp', mmin, mchar_min, mchar_mmax, and b should be provided")
                self.mmin = mmin
                self.mchar_min = mchar_min
                self.mchar_max = mchar_max
                self.mchar_avg = mchar_avg
                self.mchar_stdev = mchar_stdev
            case 'const':
                if mmin is None or mchar_min is None or mchar_max is None:
                    raise ValueError("For model 'const', mmin, mchar_min, mchar_max should be provided. Mag range = [mchar_avg - dm/2, mchar_avg + dm/2]")
                self.mmin = mmin
                self.mchar_min = mchar_min
                self.mchar_max = mchar_max
            case _:
                raise ValueError("Invalid model: 'exp', 'YC', 'PureChar', or 'const'")


    def cdf(self, x):
        x = np.array(x)
        match self.model:
            case 'exp':
                beta = self.b*np.log(10)
                denom = 1 - np.exp( -beta * (self.mmax - self.mmin) )
                numer = 1 - np.exp(-beta * (x - self.mmin))
                Fm = numer / denom
                Fm = np.where( (x <= self.mmin), 0, np.where( x>=self.mmax, 1, Fm ) ) #  & (x < self.mmax)
            case 'yc':
                Fm = np.zeros_like(x)

                mask_exp = x<self.mchar_min
                mask_char = ~mask_exp
                
                x_exp = x[mask_exp]
                x_char = x[mask_char]

                expmodel = MagProb(model='exp', b=self.b, mmax=self.mchar_min, mmin=self.mmin)
                Fm[mask_exp] = expmodel.cdf(x_exp)

                slopeCDF_char = expmodel.pdf(self.mchar_min-self.dm_equi)
                Fm[mask_char] = expmodel.cdf(self.mchar_min) + slopeCDF_char*(x_char-self.mchar_min)
                
                norm_factor = expmodel.cdf(self.mchar_min) + slopeCDF_char*(self.mchar_max-self.mchar_min)
                Fm /= norm_factor

                Fm = np.where( (x < self.mmin), 0, np.where(x>self.mchar_max, 1, Fm) )
                
            case 'purechar':
                a_scaled = (self.mchar_min - self.mchar_avg) / self.mchar_stdev
                b_scaled = (self.mchar_max - self.mchar_avg) / self.mchar_stdev
                probmodel = truncnorm(a_scaled, b_scaled, loc=self.mchar_avg, scale = self.mchar_stdev)
                Fm = probmodel.cdf(x)

            case 'const':
                Fm = (x-self.mchar_min) / (self.mchar_max - self.mchar_min)
                Fm = np.where(x<self.mmin, 0, np.where( (x < self.mchar_min), 0, np.where(x>self.mchar_max, 1, Fm) ))

        return Fm


    def pdf(self, x):
        x = np.array(x)
        match self.model:
            case 'exp':
                beta = self.b*np.log(10)
                denom = 1 - np.exp( -beta * (self.mmax - self.mmin) )
                numer = beta * np.exp(-beta * (x - self.mmin))
                fm = numer / denom
                fm = np.where((x >= self.mmin) & (x <= self.mmax), fm, 0)
            case 'yc':
                fm = np.zeros_like(x)
                mask_exp = x<self.mchar_min
                mask_char = ~mask_exp
                
                #### Exp range ####
                x_exp = x[mask_exp]
                expmodel = MagProb(model='exp', b=self.b, mmax=self.mchar_min, mmin=self.mmin)
                fm[mask_exp] = expmodel.pdf(x_exp)

                #### Char range #####
                slopeCDF_char = expmodel.pdf(self.mchar_min-self.dm_equi)
                fm[mask_char] = slopeCDF_char
                
                ### norm factor ###
                norm_factor = expmodel.cdf(self.mchar_min) + slopeCDF_char*(self.mchar_max-self.mchar_min)

                fm /= norm_factor
                fm = np.where((x > self.mmin) & (x < self.mchar_max), fm, 0)
                
            case 'purechar':
                a_scaled = (self.mchar_min - self.mchar_avg) / self.mchar_stdev
                b_scaled = (self.mchar_max - self.mchar_avg) / self.mchar_stdev
                probmodel = truncnorm(a_scaled, b_scaled, loc=self.mchar_avg, scale = self.mchar_stdev)
                fm = probmodel.pdf(x)
            
            case 'const':
                fm = 1 / (self.mchar_max - self.mchar_min)
                fm = np.where( (x>self.mmin) & (x > self.mchar_min) & (x < self.mchar_max), fm, 0 )

        return fm
    
    def inverse_cdf(self, u):
        """ Numerical inversion of CDF using interpolation """
        match self.model:
            case 'exp':
                beta = self.b*np.log(10)
                denom = 1 - np.exp( -beta * (self.mmax - self.mmin) )
                m = np.log(1 - denom*u)/(-beta) + self.mmin
                return m
                #x_vals = np.linspace(self.mmin, self.mmax, N)  # Dense grid
            case 'yc':
                interval = self.mchar_max - self.mmin
                N = int(interval / 0.00001)
                x_vals = np.linspace(self.mmin, self.mchar_max, N)  # Dense grid
            case 'purechar':
                raise ValueError("model = 'exp', 'YC', 'PureChar'")
        cdf_vals = self.cdf(x_vals)
        return np.interp(u, cdf_vals, x_vals)  # Inverse interpolation
    
    def sample(self, size=1):
        """ Generate random samples using inverse transform sampling """
        match self.model:
            case 'exp' | 'yc':  # Both require inverse transform sampling
                u = np.random.uniform(0, 1, size)
                return self.inverse_cdf(u)

            case 'purechar':  # Direct sampling from truncated normal
                a_scaled = (self.mchar_min - self.mchar_avg) / self.mchar_stdev
                b_scaled = (self.mchar_max - self.mchar_avg) / self.mchar_stdev
                probmodel = truncnorm(a_scaled, b_scaled, loc=self.mchar_avg, scale=self.mchar_stdev)
                return probmodel.rvs(size=size)
            
            case 'const':
                return np.random.uniform(low=self.mchar_min, high=self.mchar_max, size=size)

