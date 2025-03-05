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
psha = PSHA_singlesource(PEER_TEST_1_11, GMi_list = [0.001, 0.1, 1])
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
