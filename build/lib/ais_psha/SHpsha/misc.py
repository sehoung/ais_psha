import numpy as np
def xyz2dist(site, xyz):
    xyz = np.array(xyz)
    site = np.array(site)
    
    x,y,z = xyz.T
    x0, y0, z0 = site.T

    r = ((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**0.5
    return r