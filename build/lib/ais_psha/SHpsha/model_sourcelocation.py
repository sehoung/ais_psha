import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
class PointSourceModel:
    def __init__(self, loc = [10, 10, 5]):
        self.loc = np.array(loc)

    def pdf(self, X):
        pdfval = np.ones(X.shape[0])/len(X)
        return pdfval

    def sample(self, Nsmpl=1):
        return np.tile(self.loc, (Nsmpl,1))


class PolygonSourceModel:
    def __init__(self, coordinate=[[100,0], [0,0], [0, 100], [-30,30], [-20,0], [-100,0], [50,-20],[0, -100]], depthrange=[5,10], depth_distribution='uniform'):
        
        self.depthrange = depthrange
        self.distribution = depth_distribution
        self.z1=self.depthrange[0]
        self.z2=self.depthrange[1]
        self.length = (self.z2-self.z1)
        
        # Ensure the polygon is closed by appending the first point to the end
        if not np.array_equal(coordinate[0], coordinate[-1]):
            coordinate = np.vstack([coordinate, coordinate[0]])
        self.coordinate = coordinate

    def pdf(self, X):
        x,y,z = X.T

        ### pdf of x and y ###
        valid_range = self.is_point_inside_polygon(np.c_[x,y])
        area = self.compute_polygon_area()

        pdf = np.zeros(len(x))
        pdf[valid_range] = 1/area
        
        ### pdf of z (depth) ###
        match self.distribution:
            case 'uniform':
                #pdf_z = np.zeros(len(z))
                pdf_z = np.where(  (self.z1<=z) & (z<=self.z2), 1./self.length, 0  )

            case 'triangle':
                center = (self.depthrange[1]+self.depthrange[0])/2.
                height = 1/(0.5*self.length)
                slope1 = height / (center-self.z1)
                slope2 = -slope1
                pdf_z = np.where(  (self.z1<z) & (z<center), slope1*(z-self.z1), np.where(  (center<=z) & (z<self.z2), slope2*(z-self.z2), 0  )     )

        pdf *= pdf_z
        return pdf

    def sample(self, Nsmpl=1):
        Nsmpl_init=Nsmpl

        #### Sample x and y ####
        coord = np.array(self.coordinate)
        x_a = np.min(coord[:,0])
        x_b = np.max(coord[:,0])
        y_a = np.min(coord[:,1])
        y_b = np.max(coord[:,1])

        tf = False
        i=0
        while np.any(tf==False):
            xs = np.random.uniform(x_a, x_b, size=Nsmpl)
            ys = np.random.uniform(y_a, y_b, size=Nsmpl)

            xy = np.c_[xs, ys]

            tf = np.array(self.is_point_inside_polygon(xy))
            
            if i==0:
                xy_sel = xy[tf].T
            else:
                xy_sel = np.c_[xy_sel, xy[tf].T]
            
            Nsmpl = len(xy[~tf])
            
            i+=1

        ### sample z (depth) ###
        Nsmpl = Nsmpl_init
        match self.distribution:
            case 'uniform':
                z = np.random.uniform(low=self.z1, high=self.z2, size=Nsmpl)

            case 'triangle': #################  MODIFY! CDF IS NOT CORRECRT!!
                y = np.random.uniform(low=0, high=1, size=Nsmpl)
                center = (self.depthrange[1]+self.depthrange[0])/2.
                height = 1/(0.5*self.length)
                slope1 = height / (center-self.z1)
                slope2 = -slope1
                #samples = np.where(  (0<=y) & (y<0.5), (self.z1*slope1 + np.sqrt((self.z1*slope1)**2 + 2*slope1*y))/slope1     , (self.z2*slope2 + np.sqrt((self.z2*slope2)**2 + 2*slope2*y))/slope2 )
                #samples = np.where(  (0<=y) & (y<0.5), (self.z1 - np.sqrt((self.z1)**2 - 2/slope1*y)) , self.z2 - np.sqrt((self.z2)**2 - 2/slope2*y) ) 
                #samples = np.where(  (0<=y) & (y<0.5), self.z1 + (self.z2-self.z1)*np.sqrt(y) , self.z2 - (self.z2-self.z1)*np.sqrt(1-y) ) 
                #samples = np.where(  (0<=y) & (y<0.5), self.z1 + np.sqrt(2*y/slope1), self.z2 - np.sqrt(2*(1-y)/slope2) ) 
                z = np.where(  (0<=y) & (y<0.5), self.z1 + np.sqrt( slope1**2*self.z1**2 + 2* slope1*y )  / (slope1)
                                                        , self.z2 + np.sqrt( slope2**2*self.z2**2 + 2* slope2*y )  / (slope2) ) 

        return np.c_[xy_sel.T, z]

    def is_point_inside_polygon(self, points):
        vertices = self.coordinate
        
        codes = [Path.LINETO]* (len(vertices)-2)
        codes.insert(0,Path.MOVETO)
        codes.append(Path.CLOSEPOLY)

        # Create a Path object
        path = Path(vertices, codes)

        # Check if the path contains the points
        inside = path.contains_points(points)
        return inside

    def compute_polygon_area(self):
        """
        Computes the area of a 2D polygon using the Shoelace formula.

        Parameters:
            coordinates (list of tuples): A list of (x, y) tuples representing the polygon's vertices.

        Returns:
            float: The area of the polygon.
        """
        # Convert the list of tuples into a NumPy array for easier manipulation
        coords = np.array(self.coordinate)
        
        ## Ensure the polygon is closed by appending the first point to the end
        #if not np.array_equal(coords[0], coords[-1]):
        #    coords = np.vstack([coords, coords[0]])
        
        # Extract x and y coordinates
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Apply the Shoelace formula
        area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
        return area

########## ON TEST 03/05/2025. SHOULD BE MODIFIED IN THE FUTURE ################
class FaultSourceModel:
    def __init__(self, MagScalingModel, fault_coord = [[0, 12.5, 0], [0, 12.5, 12], [0, -12.5, 12], [0, -12.5, 0]]):
        self.coord = fault_coord
        self.MagScalingModel = MagScalingModel

    def pdf(self, X):
        pdfval = np.ones(X.shape[0])
        return pdfval

    def sample(self, Nsmpl=1):
        return np.tile(self.loc, (Nsmpl,1))


'''circle_points = [
    [ 100.000000,  0.000000],
    [ 99.452190,  10.452846],
    [ 97.814760,  20.791169],
    [ 95.105652,  30.901699],
    [ 91.354546,  40.673664],
    [ 86.602540,  50.000000],
    [ 80.901699,  58.778525],
    [ 74.314483,  66.913061],
    [ 66.913061,  74.314483],
    [ 58.778525,  80.901699],
    [ 50.000000,  86.602540],
    [ 40.673664,  91.354546],
    [ 30.901699,  95.105652],
    [ 20.791169,  97.814760],
    [ 10.452846,  99.452190],
    [  0.000000, 100.000000],
    [-10.452846,  99.452190],
    [-20.791169,  97.814760],
    [-30.901699,  95.105652],
    [-40.673664,  91.354546],
    [-50.000000,  86.602540],
    [-58.778525,  80.901699],
    [-66.913061,  74.314483],
    [-74.314483,  66.913061],
    [-80.901699,  58.778525],
    [-86.602540,  50.000000],
    [-91.354546,  40.673664],
    [-95.105652,  30.901699],
    [-97.814760,  20.791169],
    [-99.452190,  10.452846],
    [-100.000000,  0.000000],
    [-99.452190, -10.452846],
    [-97.814760, -20.791169],
    [-95.105652, -30.901699],
    [-91.354546, -40.673664],
    [-86.602540, -50.000000],
    [-80.901699, -58.778525],
    [-74.314483, -66.913061],
    [-66.913061, -74.314483],
    [-58.778525, -80.901699],
    [-50.000000, -86.602540],
    [-40.673664, -91.354546],
    [-30.901699, -95.105652],
    [-20.791169, -97.814760],
    [-10.452846, -99.452190],
    [ -0.000000, -100.000000],
    [ 10.452846, -99.452190],
    [ 20.791169, -97.814760],
    [ 30.901699, -95.105652],
    [ 40.673664, -91.354546],
    [ 50.000000, -86.602540],
    [ 58.778525, -80.901699],
    [ 66.913061, -74.314483],
    [ 74.314483, -66.913061],
    [ 80.901699, -58.778525],
    [ 86.602540, -50.000000],
    [ 91.354546, -40.673664],
    [ 95.105652, -30.901699],
    [ 97.814760, -20.791169],
    [ 99.452190, -10.452846]
]'''

