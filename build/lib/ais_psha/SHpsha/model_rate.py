import numpy as np
from scipy.integrate import quad
import warnings

class FaultEQRate():
    def __init__(self, MagProbModel, Area=None, fault_coord = None, sliprate=1):
        """
        fault_coord = in km
        Area = km2
        slip rate = mm/yr
        """
        if Area is not None and fault_coord is not None:
            raise ValueError("Only one of 'Area' or 'fault_coord' can be defined, not both.")        
        elif Area is not None and fault_coord is None:
            self.Area_km2 = Area
        elif Area is None and fault_coord is not None:
            self.fault_coord = fault_coord
            self.Area_km2 = self.fault_area()

        self.Area_cm2 = self.Area_km2*10**10 # convert area to cm2
        
        self.sliprate_mmyr = sliprate
        self.sliprate_cmyr = sliprate * 0.1 # convert mm/yr to cm/yr

        self.MagProbModel = MagProbModel
        
        self.M0rate = self.calc_M0rate()
        self.rate = self.calc_rate_Mmin()
        
    def moment_release(self, M):
        moment = 10**(1.5*M + 16.05)
        pdf = self.MagProbModel.pdf(M)
        return  moment*pdf

    def mean_moment_per_eq(self):
        def integrand(M):
            moment = 10**(1.5*M + 16.05)
            pdf = self.MagProbModel.pdf(M)
            return  moment*pdf
        integ, err = quad(integrand, 0, 10)
        return integ

    def calc_rate_Mmin(self):
        M0rate_per_eq = self.mean_moment_per_eq()
        
        rate_M_gt_0 = self.M0rate / M0rate_per_eq
        a = self.MagProbModel.cdf(self.MagProbModel.mmin)
        rate = (1-a) * rate_M_gt_0

        return rate
    
    def calc_M0rate(self):
        # A and S in cm
        # output is dyne-cm/year
        mu = 3e11 # dyne/cm2
        return mu * self.Area_cm2 * self.sliprate_cmyr
    
    def fault_area(self):
        """
        Calculate the area of a rectangle in 3D space given its four vertices.
        
        Parameters:
            coords (numpy.ndarray): 4x3 array containing the four (x, y, z) coordinates. it should be in drawing order
        
        Returns:
            float: The area of the rectangle.
        """
        coords = np.array(self.fault_coord)

        if coords.shape != (4, 3):
            raise ValueError("Input must be a 4x3 array of 3D coordinates.")

        # Compute two adjacent edge vectors
        vec1 = coords[1] - coords[0]  # horizontal
        vec2 = coords[3] - coords[0]  # vertical
        area1 = np.linalg.norm(np.cross(vec1, vec2))

        vec1 = coords[3] - coords[2]  
        vec2 = coords[1] - coords[2]  
        area2 = np.linalg.norm(np.cross(vec1, vec2))

        vec1 = coords[1] - coords[0]  
        vec2 = coords[1] - coords[2]  
        area3 = np.linalg.norm(np.cross(vec1, vec2))

        vec1 = coords[3] - coords[2]  
        vec2 = coords[3] - coords[0]  
        area4 = np.linalg.norm(np.cross(vec1, vec2))

        
        area = np.mean([area1, area2, area3, area4])
        cov = np.std([area1, area2, area3, area4]) / area
        if cov>0.01:
            print("cov", cov)
            warnings.warn("The fault coordinates are not coplanar (they does NOT lie on the same plane!)")
        return area
    

    #def calc_charM(self):
    #    return np.log10(self.Area_km2)+4