from ais_psha.SHpsha import MagProb, PolygonSourceModel, GroundMotionModel, PSHA_singlesource
import numpy as np


### Define Source and Ground Motion Models ###
r = 100
th = np.arange(0, 360, 6)
circle_points = np.c_[r*np.cos(th*np.pi/180), r*np.sin(th*np.pi/180)]

PEER_TEST_1_11 = {
    1: {"name": "src1",
        "rate" : 0.0395,
        "m": MagProb(model='exp', mmin=5.0, mmax=6.5, b=0.9),
        "loc": PolygonSourceModel(coordinate=circle_points),
        "gmm": GroundMotionModel(model='sadigh1997', sample_eps=False),
        },
}

GMi_list = np.array([0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1])

psha = PSHA_singlesource(PEER_TEST_1_11, GMi_list = GMi_list)
###############################################

# Define integration range for AIS (magnitude, x coordinate, y coordintae, and z coordinate) 
integ_range = np.array([[0, 10], [-100, 100], [-100, 100], [0, 20]])


##### Run Hazard #####
# Run naive MC hazard
haz, cov = psha.haz(Nhazsmpl = 100000, method="MC")
print(haz, cov)

# Run Population Monte Carlo AIS hazard
haz, cov = psha.haz(Nhazsmpl = 10000, method="PMC", integ_range = integ_range)
print(haz, cov)

# Run VEGAS AIS hazard
haz, cov = psha.haz(Nhazsmpl = 10000, method="vegas", integ_range = integ_range)
print(haz, cov)
########################
