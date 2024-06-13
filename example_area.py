from configuration_area.integ_setting import *

from include.example_psha import PSHA_area1
from include.vegas_v1 import Run_vegas_integ
from include.vegas_misc import Run_vegas_figure
import numpy as np

GMi = 1.0 # targe ground motion in g

integ_range = np.array( [ [Mmin_integ, Mmax_integ], [Dmin_integ, Dmax_integ], [zmin_integ,zmax_integ],  [emin_integ, emax_integ]] ) # integration range
Nsmpl = 100000 # number of samples
Ngrid = np.full(len(integ_range), 50) # number of grids
Niter = 10 # maximum iteration

### RUN PSHA ###
Bdr_hist, cont_hist, integ_hist, CV_hist, sus_flag = Run_vegas_integ(PSHA_area1, integ_range, Ngrid, Nsmpl, Niter, f_arg = [GMi]) # run AIS VEGAS PSHA
Run_vegas_figure(Bdr_hist, integ_range, integ_hist, CV_hist, sus_flag, Ngrid)