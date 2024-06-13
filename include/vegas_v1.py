import numpy as np
import math
import random
from pprint import pprint
import matplotlib.pyplot as plt
import sys
import time
from scipy import stats

def Run_vegas_integ(f, integ_range, Ngrid, Nsmpl, Niter, f_arg=[]):
    ##  ERROR ##
    if len(integ_range) != len(Ngrid):
        print("ERROR: INPUT MISMATCH : The lengths of lists integ_range and Ngrid should be identical.", file=sys.stderr)
        return [0],[0],[0]
    
    ## Initialize
    cont_hist = {}
    cont_damp_hist = {}
    Bdr_hist = {}
    integ_hist = np.array([])
    CV_hist=np.array([])
    Bdr, Len, qx = Init_grid(integ_range, Ngrid)
    
    ## Main loop
    for k in range(0, Niter+1, 1):
        smpl, smplidx, smplqx = Sample_VEGAS_grid(Bdr, qx, Ngrid, Nsmpl) # 1. Sampling
        fx = f(np.array(smpl), arg=f_arg) # 2. Run function
        cont, cont_damp, cont_hist, cont_damp_hist = Calc_contribution(fx, qx, Bdr, Len, Ngrid, smpl, smplidx, smplqx, cont_hist, cont_damp_hist, k) # 3. Compute contribution

        if np.sum(cont[0]) <0.1: # ERROR: np.sum(cont[0]) should be ONE if the cont was properly computed
            integ_hist, CV_hist = calc_integ_CV(fx, Nsmpl, smpl, smplidx, qx, integ_range, integ_hist, CV_hist, k)
            print_result(k, integ_hist, CV_hist, flag=-2)
            return Bdr_hist, cont_hist, integ_hist, CV_hist -1
        else:
            integ_hist, CV_hist = calc_integ_CV(fx, Nsmpl, smpl, smplidx, qx, integ_range, integ_hist, CV_hist, k) # 4. Compute Integration
            Bdr, Len, qx, Bdr_hist = Update_grid(Bdr, qx, Len, Ngrid, cont_damp, Bdr_hist, k) #5. Update Grid
 
            if k>0 and CV_hist[-1] > CV_hist[-2] and CV_hist[-2]<0.2 : # NORMAL
                print_result(k, integ_hist, CV_hist, flag=0)
                return Bdr_hist, cont_hist, integ_hist, CV_hist, 1
        
            if k==Niter: # MAX ITER REACHED
                print_result(k, integ_hist, CV_hist, flag=-1)
                return Bdr_hist, cont_hist, integ_hist, CV_hist, -1
#################################################################

def print_result(k, integ_hist, CV_hist, flag=0):
    formatting_percent = np.vectorize(lambda x: f"{x:.2f}")
    CV_hist_pct = formatting_percent(CV_hist*100)
    if flag == 0: # Normal termination
        print("stopped at k = ", k, "," , "Integ = ", "{:.12e}".format(integ_hist[-2]), "," ,  "CV = ", CV_hist_pct[-2], "%", file=sys.stderr)
        return 0
    elif flag == -1:
        print("WARNING: stopped at k = ", k, "," , "Integ = ", "{:.12e}".format(integ_hist[-1]), "," ,  "CV = ", CV_hist_pct[-1], "%", "(MAX ITER REACHED)", file=sys.stderr)
        return -1
    elif flag== -2:
        print("ERROR: stopped at k = ", k, "," , "Integ = ", "{:.12e}".format(integ_hist[-1]), "," ,  "CV = ", CV_hist_pct[-1], "%", "(GRID CANNOT BE UPDATED)", file=sys.stderr)


def Init_grid(range, Ngrid):
    Ndim = len(range)
    Boundary = {}
    Gridlen = {}
    qx = {}
    for i in np.arange(Ndim):
        a = range[i][0]
        b = range[i][1]
        N = Ngrid[i]
        key = int(i)        
        Boundary[key] = np.arange( a, b + ((b-a)/N/2), (b-a)/N )
        Gridlen[key] = np.diff(Boundary[key])
        qx[key] = np.full( N , 1/(b-a) ) 
    return Boundary, Gridlen, qx


def Sample_VEGAS_grid(Boundary, qx, Ngrid, Nsmpl):
    Ndim = len(Ngrid)
    smplqx = np.zeros((Nsmpl, Ndim))
    smplidx = np.zeros((Nsmpl, Ndim)).astype(int)
    smpl = np.zeros((Nsmpl, Ndim))

    for i_dim, N in enumerate(Ngrid):
        smplidx[:,i_dim] = np.random.randint(0, N, size=Nsmpl) ## Pick random VEGAS grid
        idx = smplidx[:,i_dim].astype(int)
        
        start_points = Boundary[i_dim][idx]
        end_points = Boundary[i_dim][idx + 1]
        smpl[:,i_dim] = np.random.uniform(start_points, end_points) # Generate random numbers within the grid
        smplqx[:,i_dim] = qx[i_dim][ smplidx[:,i_dim] ] 
    
    return smpl, smplidx, smplqx

def Sample_IS_grid(Boundary, Ngrid, Nsmpl, pdf):
    # pdf : dictionary
    Ndim = len(Ngrid)
    smplidx = np.zeros((Nsmpl, Ndim))
    smpl = np.zeros((Nsmpl, Ndim))
    
    for i, N in enumerate(Ngrid):
        smplidx[:,i] = np.random.randint(0, N, size=Nsmpl) ## Pick random VEGAS grid
        idx = smplidx[:,i].astype(int)
        
        start_points = Boundary[i][idx]
        end_points = Boundary[i][idx + 1]
        smpl[:,i] = np.random.uniform(start_points, end_points) # Generate random numbers within the grid
        
    return smpl, smplidx

def calc_IS_integ(f, Nsmpl, smpl, smplidx, qx, integ_range):
    integ = 0
    Ndim = len(smpl.T)
    ith_smpl_ith_dim_grid = smplidx.astype(int)
    
    integ_range = np.array(integ_range)
    range_widths = integ_range[:, 1] - integ_range[:, 0]
    pdf_init = 1 / range_widths
    
    wt_array = np.ones(Nsmpl)
    for i_dim in range(Ndim):
        pdf_learn_values = qx[i_dim][ith_smpl_ith_dim_grid[:, i_dim]]
        wt_array *= pdf_init[i_dim] / pdf_learn_values
    integ_frac=np.zeros(Nsmpl)

    integ_frac = f(np.array(smpl))*wt_array
    integ = np.sum(integ_frac)

    integ_vol = 1
    for i_dim in range(Ndim):
        integ_vol *= integ_range[i_dim][1] - integ_range[i_dim][0]
    integ=integ_vol*integ/Nsmpl

    return integ
#########################################


def calc_integ_CV(fx, Nsmpl, smpl, smplidx, qx, integ_range, integ_hist, CV_hist, kth_iter):
    integ = 0
    Ndim = len(smpl.T)
    ith_smpl_ith_dim_grid = smplidx.astype(int)
    
    integ_range = np.array(integ_range)

    ## compute original density ##
    range_widths = integ_range[:, 1] - integ_range[:, 0]
    pdf_init = 1 / range_widths
    
    ## compute importance weight ##
    wt_array = np.ones(Nsmpl)
    for i_dim in range(Ndim):
        pdf_learn_values = qx[i_dim][ith_smpl_ith_dim_grid[:, i_dim]]
        wt_array *= pdf_init[i_dim] / pdf_learn_values
    integ_frac=np.zeros(Nsmpl)


    ## compute integration volume ##
    integ_vol = 1
    for i_dim in range(Ndim):
        integ_vol *= integ_range[i_dim][1] - integ_range[i_dim][0]

    ## compute integration ##
    integ_frac = fx*wt_array*integ_vol
    integ = np.mean(integ_frac)

    ## compute COV (CV) ##
    integ_frac_sq = integ_frac * integ_frac
    integ_frac_sq_mean = np.mean(integ_frac_sq)
    VAR = ( integ_frac_sq_mean - integ**2) / Nsmpl
    if integ ==0:
        CV=0
    else:
        CV = VAR**0.5 / integ
    ##################

    integ_hist = np.append(integ_hist, integ)
    CV_hist = np.append(CV_hist, CV)

    return integ_hist, CV_hist
#########################################

def Calc_contribution(fx, qx, bdr, Len, Ngrid, smpl, smplidx, smplqx, d_hist, d_damp_hist, kth_iter, alpha = 1.0):
    optimizer_frac = np.zeros_like(smpl)
    fx_sq = np.zeros(len(smpl))
    d = qx.copy()
    d = {key: np.zeros_like(value) for key, value in d.items()}
    d_damp = {key: np.zeros_like(value) for key, value in d.items()}

    ## 1. CALCULATING ACTUAL CONTRIBUTION
    fx_sq = fx*fx
    if np.sum(fx_sq)==0:
        print("ERROR: Too small functional values!", file=sys.stderr)
        d_hist[kth_iter] = d.copy()
        d_damp_hist[kth_iter] = d_damp.copy()
        return d, d_damp, d_hist, d_damp_hist
    
    for i_dim in d: ## run for dimension
        smplqx_i_ne_j = np.delete(smplqx, i_dim, axis=1)
        optimizer_frac[:,i_dim] = np.prod(smplqx_i_ne_j, axis=1) * fx_sq # fractional optimizer for i_dim
        idx_i_dim = smplidx[:,i_dim]
        for i_grid in range(Ngrid[i_dim]):
            idx = np.where(idx_i_dim==i_grid)[0]
            d[i_dim][i_grid] =np.sum( optimizer_frac[ idx, i_dim] )
        d[i_dim] /= Len[i_dim]
        d[i_dim][:] = np.sqrt(d[i_dim][:])
    
    for i_dim in d: # normalize d
        d[i_dim][:] /=np.sum(d[i_dim][:])
        
    ## 2. SMOOTHING CONTRIBUTION
    for i_dim in d:
        for i in range(Ngrid[i_dim]):
            if i == 0:
                d_damp[i_dim][i] = ( 7*d[i_dim][0] + d[i_dim][1] ) / 8.
            elif i == Ngrid[i_dim]-1:
                d_damp[i_dim][i] = ( d[i_dim][Ngrid[i_dim]-2] + 7*d[i_dim][Ngrid[i_dim]-1] ) / 8.
            else:
                d_damp[i_dim][i] = ( d[i_dim][i-1] + 6*d[i_dim][i] + d[i_dim][i+1] ) / 8.
        
        for i in range(Ngrid[i_dim]):
            if d_damp[i_dim][i] > 0:
                A = (1-d_damp[i_dim][i])
                B = ( -1*math.log(d_damp[i_dim][i]) )
                logX = alpha* ( math.log(A) - math.log(B))
                d_damp[i_dim][i] = math.exp(logX)
                #d_damp[i_dim][i] = ( (1-d_damp[i_dim][i]) / ( -math.log(d_damp[i_dim][i]) ) )**alpha
            else:
                d_damp[i_dim][i] = 0
        d_damp[i_dim]/=np.sum(d_damp[i_dim])
    
    d_hist[kth_iter] = d.copy()
    d_damp_hist[kth_iter] = d_damp.copy()

    return d, d_damp, d_hist, d_damp_hist


def Update_grid(Boundary, qx, Len, Ngrid, d, Boundary_hist, kth_iter, K=1000):      
    ## 1. CALCULATE # OF SUBGRID PER GRID
    M = {key: np.round(d[key] * K) for key in d} 
    for i_dim in M: ## prevent any number M to be 0
        for i, Mval in enumerate(M[i_dim]):
            if M[i_dim][i] == 0:
                M[i_dim][i]=1
        while K - np.sum(M[i_dim][:-1]) < 0.5:
            M[i_dim]=subtract_one_random_row(M[i_dim]) 
        M[i_dim][-1] = K - np.sum(M[i_dim][:-1])

    ## 2. SUBDIVIDE AND MERGE BOUNDARY 
    subgrid_boundaries = {key: [] for key in Boundary} 
    for i_dim in Boundary:
        for i in range(len(M[i_dim])):
            subgrid_step = (Boundary[i_dim][i + 1] - Boundary[i_dim][i]) / M[i_dim][i]
            ##print(M[i_dim][i]) #################
            subgrid_boundaries[i_dim].extend([Boundary[i_dim][i] + subgrid_step * j for j in np.arange(M[i_dim][i])])
        ## Add the last boundary point of the last grid
        subgrid_boundaries[i_dim].append(Boundary[i_dim][-1])
        Boundary[i_dim] = subgrid_boundaries[i_dim][::int(K/Ngrid[i_dim])]
    
    Len = {key: np.diff(Boundary[key]) for key in Boundary}
    for i_dim in Boundary:
        for j in range(Ngrid[i_dim]):
            qx[i_dim][j] = 1 / ( Ngrid[i_dim]*Len[i_dim][j] )
    
    ### Convert list to numpy array. This is mandatory for the use in Sample_VEGAS
    for key in Boundary:
        Boundary[key] = np.array(Boundary[key])

    Boundary_hist[kth_iter] = Boundary.copy()

    return Boundary, Len, qx, Boundary_hist
#########################################


def subtract_one_random_row(arr):
    value=1.5
    eligible_indices = [i for i, element in enumerate(arr) if element > value]

    if eligible_indices:
        random_index = random.choice(eligible_indices)
        arr[random_index] -= 1
        return arr
    else:
        print("No elements with values greater than", value, "found in the array.")
        return None
###################################################