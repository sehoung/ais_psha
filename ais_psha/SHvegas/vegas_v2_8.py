import numpy as np
import math
import random
from pprint import pprint
import matplotlib.pyplot as plt
import sys
import time
from scipy import stats
import copy

# Fit a simpler function to the KDE result (optional)

from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator
from collections import namedtuple


##### VEGAS version 2 #####
# The normally distributed initial guess can be implemented #
##### VEGAS version 2_1 #####
# calc_contribution function is vectorized for improving the computational efficiency. ~10% faster when N_dim=9, Ngrid=10*9
# For Ndim=9, Ngrid=50*9, it is ~25% faster
##### VEGAS version 2_2 #####
# 1. correc error from version 2 and 2_1: in case of normal as initial distribution, corrected the importance weight calculation part
# -> added to the sample_vegas_grid and modification on integ function
# but still not sure whether it works or not
# 2. Bdr_hist is modified
##### VEGAS version 2_3 ######
# output includes qx_hist
#### VEGAS version 2_4 ######
# add LHS in vegas : not efficient than naive sampling...
# add smoothing of qx
###### VEGAS version 2_5 ######
# MINOR
###### VEGAS version 2_6 ######
# update_grid IS vectorized
# optimizer midified
# Now it converges
# Stopping criteria: when CV change not much
### VEGAS version 2_7 ######
# calc q_star
# add output: smpl, q_star
# function explanation is added 
# funciton marginalize_pdf is added
### VEGAS version 2_7_mod ######
## trial q section is added ##
## new stopping criteria: cont_damp< 10% -> stop ##
### VEGAS version 2_7_mod2 ######
## trial q section stoppinc criteria changed ##
### VEGAS version 2_8
## Add options for stopping criteria ###
## Should add CV criteria ##


VEGASoutput = namedtuple('VEGASoutput', [
    'boundary', 'qx', 'contribution', 'integration', 'COV',
    'Kopt', 'flag', 'sample', 'q_star_sample', 'sample_init',
    'q_star_sample_init', 'sample_all', 'q_star_sample_all', 'smplidx', 'Len'])
VEGASoutput.__new__.__defaults__ = (
    [0],  # boundary
    0.0,   # qx
    0.0,   # contribution
    0.0,   # integration
    0.0,   # COV
    0,  # Kopt
    -1, # flag
    [0],    # sample
    [0],    # q_star_sample
    [0],    # sample_init
    [0],    # q_star_sample_init
    [0],    # sample_all
    [0],    # q_star_sample_all
    0,     # smplidx
    0      # Len
)

SUBGRID_MUTIPLIER = 200

def Run_vegas_integ(f, integ_range, Ngrid, Nsmpl,
                    Niter=50,
                    alpha=1.0,
                    initial_guess='uniform',
                    initial_guess_param=[],
                    f_arg=[],
                    sampling_method = 'none',
                    N_resample = 10,
                    term_condition = 'cont', ## default, CV
                    CV_threshold=0.01,
                    cont_CV_change_threshold = 0.20,
                    print_output = True):
    """
    f: target integrand (function)
        BE CLEAR that the FUNCTION OUTPUT MUST be 1-D ARRAY.
        Note that the function input is always 2-d array with the shape (Nsmpl*Ndim).
        That is, even in 1-dimensional case, the input sample is (Nsmpl*1) 2-d array.
        To manage this I recommend that your function start with x = X[:,0] in 1-d case, x1, x2, x3 ... = X[:,0], X[:,1], X[:,2], ... in N-dimensional case.
    integ_range: 2-d array with shape (Ndim * 2). e.g., [[-10,10],[0,5], [-7,2]]
    Ngrid: 1-d array with shape Ndim. e.g., [50, 50, 50]
    Nsmpl: floating number

    **IMPORTANT: The efficiency of VEGAS totally depends on the number of sub-grids. If you feel that the algorithm is TOO SLOW, reduce SUBGRID_MULTIPLIER
    """

    ### CHECK INPUT DATA ###
    if check_input(integ_range, Ngrid) == -1:
        return VEGASoutput()
    ##########################
    
    ## 1. NORMAL VEGAS ALGORITHM ##
    ## 1-a) Initialize 
    Bdr_hist = {}
    qx_hist={}
    cont_hist = {}
    cont_damp_hist = {}
    integ_hist = np.array([])
    CV_hist=np.array([])
    cont_COV_all_hist = np.array([])
    Bdr, Len, qx = Init_grid(integ_range, Ngrid, initial_guess, initial_guess_param)
    
    Bdr0, Bdr_hist[0] = copy.deepcopy(Bdr), copy.deepcopy(Bdr)
    qx0, qx_hist[0] = copy.deepcopy(qx), copy.deepcopy(qx)    

    if sampling_method == 'LHS':
        Nsmpl = Ngrid[0]

    ## 2-b) Main loop
    for k in range(0, Niter+1, 1):
        for i in range(N_resample):
            smpl, smplidx, smplqx, smplqx0 = Sample_VEGAS_grid(Bdr, Bdr0, qx, qx0, Ngrid, Nsmpl, sampling_method) # 1. Sampling
            fx = f(np.array(smpl), arg=f_arg) # 2. Run function

            ### CHECK ALL ZERO FUNCTIONAL VALUE ###
            all_zero_flag = check_all_zero(fx, i, N_resample)
            if  all_zero_flag == 1:
                break
            elif all_zero_flag == -1:
                return VEGASoutput()
            ############################################


        ### SAVE SAMPLES ###
        if k == 0:
            smpl_init, smpl_all = smpl, smpl
            fx_init, fx_all = fx, fx
        else:
            smpl_all = np.vstack((smpl_all, smpl))
            fx_all = np.hstack((fx_all, fx))
        #####################

        cont, cont_damp, cont_hist, cont_damp_hist = Calc_contribution2(fx, qx, Bdr, Len, Ngrid, smpl, smplidx, smplqx, cont_hist, cont_damp_hist, k, alpha) # 3. Compute contribution

        ### CHECK WHETHER cont IS PROPERLY CMPUTED ###
        if check_cont(cont)==-1: # ABNORMAL TERMINATION: np.sum(cont[0]) should be ONE if the cont was properly computed
            return VEGASoutput()
        ###########################################
        
        integ_hist, CV_hist = calc_integ_CV(fx, Nsmpl, smpl, smplidx, qx, smplqx0, integ_range, integ_hist, CV_hist, k) # 4. Compute Integration
        Bdr, Len, qx, Bdr_hist, qx_hist = Update_grid_vectorized(Bdr, qx, Len, Ngrid, cont_damp, Bdr_hist, qx_hist, k+1) # 5. Update Grid

        ### TERMINATION CRITERIA
        cont_damp_all_dimension = np.column_stack(list(cont_damp.values()))
        cont_COV_all = np.std(cont_damp_all_dimension.flatten())/np.mean(cont_damp_all_dimension.flatten())*100
        cont_COV_all_hist = np.append(cont_COV_all_hist, cont_COV_all)
        if k>0:
            change_cont_COV = np.fabs(cont_COV_all_hist[-1] - cont_COV_all_hist[-2])/ np.fabs(cont_COV_all_hist[-1] + 1e-8)
        else:
            change_cont_COV = 9999

        #if k > 0  and CV_hist[-2] < CV_threshold and CV_change < CV_threshold*0.5 : # NORMAL TERMINATION
        #if cont_COV_all < 10: # NORMAL TERMINATION
        if term_condition == 'cont':
            termination_condition = change_cont_COV < cont_CV_change_threshold 
        elif term_condition == 'CV':
            termination_condition = CV_hist[-1] < CV_threshold

        if termination_condition: 
            k_opt = print_result(k, integ_hist, CV_hist, print_output, flag=0)
            q_star_sample, q_star_sample_init, q_star_sample_all = fx/integ_hist[k_opt], fx_init/integ_hist[k_opt], fx_all/integ_hist[k_opt]
            return VEGASoutput(Bdr_hist, qx_hist, cont_hist, integ_hist, CV_hist, k_opt, 1, smpl, q_star_sample, smpl_init, q_star_sample_init, smpl_all, q_star_sample_all, smplidx, Len)
    
        if k==Niter: # MAX ITER REACHED
            k_opt = print_result(k, integ_hist, CV_hist, print_output, flag=-1)
            q_star_sample, q_star_sample_init, q_star_sample_all = fx/integ_hist[k_opt], fx_init/integ_hist[k_opt], fx_all/integ_hist[k_opt]
            return VEGASoutput(Bdr_hist, qx_hist, cont_hist, integ_hist, CV_hist, k_opt, -1,  smpl, q_star_sample, smpl_init, q_star_sample_init, smpl_all, q_star_sample_all, smplidx, Len)
#################################################################

def check_input(integ_range, Ngrid):
    if len(integ_range) != len(Ngrid):
        print("ERROR: INPUT MISMATCH. The lengths of lists integ_range and Ngrid should be identical.", file=sys.stderr)
        return -1
    else:
        return 0
    
def check_all_zero(fx, i, N_resample):
    if np.max(np.abs(fx)) > 0: # if any one sample gives non-zero functional value, continue on the next step
        return 1
    elif i < N_resample-1:
        return 0
    elif i == N_resample-1:  # ABNORMAL TERMINATION: if all the sample gives zero functional value, terminate. bad initial guess
        print("ERROR: ALL THE MC SAMPLES GIVES ZERO FUNCTIONAL VALUE. MAY BE THE BAD INITIAL GUESS", file=sys.stderr)        
        return -1
    
def check_cont(cont):
    if np.sum(cont[0]) <0.1:
        print("ERROR: GRID CANNOT BE UPDATED", file=sys.stderr)
        return -1
    else:
        return 0

def print_result(k, integ_hist, CV_hist, print_output, flag=0):
    formatting_percent = np.vectorize(lambda x: f"{x:.2f}")
    CV_hist_pct = formatting_percent(CV_hist*100)
    if flag == 0: # Normal termination
        k_opt = k
        if print_output == True:
            print("stopped at k = ", k, ","
                , "Integ = ", "{:.12e}".format(integ_hist[-1])
                , "(idx = ", k_opt ,")"
                ,  "CV = ", CV_hist_pct[-1], "%"
                , file=sys.stderr)
        return k_opt
    elif flag == -1:
        k_opt = np.argmin(CV_hist)
        if print_output == True:
            print("WARNING: stopped at k = ", k, "," , "Integ = ", "{:.12e}".format(integ_hist[-1]), "," ,  "CV = ", CV_hist_pct[-1], "%", "(MAX ITER REACHED)", "flag=", flag, file=sys.stderr)
            print("         best estimate at k = ", k_opt, "," , "Integ = ", "{:.12e}".format(integ_hist[k_opt]), "," ,  "CV = ", CV_hist_pct[k_opt], "%", "flag=", flag, file=sys.stderr)
        return k_opt
    elif flag== -2:
        k_opt = 0
        print("ERROR: stopped at k = ", k, "," , "Integ = ", "{:.12e}".format(integ_hist[-1]), "," ,  "CV = ", CV_hist_pct[-1], "%", "(GRID CANNOT BE UPDATED)", "flag=", flag, file=sys.stderr)
        return k_opt
        
    elif flag== -3:
        k_opt = 0
        print("ERROR: stopped at k = ", k, "," , "Integ = ", 0, "," ,  "CV = ", 0, "%", "(ALL THE MC SAMPLES GIVES ZERO FUNCTIONAL VALUE. MAY BE THE BAD INITIAL GUESS)", "flag=", flag, file=sys.stderr)
        return k_opt


def Init_grid(range, Ngrid, initial_guess, initial_guess_param):
    Ndim = len(range)
    Boundary = {}
    Gridlen = {}
    qx = {}
    
    if initial_guess == 'uniform':
        for i in np.arange(Ndim):
            a = range[i][0]
            b = range[i][1]
            N = Ngrid[i]
            key = int(i)
            Boundary[key] = np.arange( a, b + ((b-a)/N/2), (b-a)/N )
            Gridlen[key] = np.diff(Boundary[key])
            qx[key] = np.full( N , 1/(b-a) )
    elif initial_guess == 'normal':
        if len(initial_guess_param) != len(initial_guess_param):
            print("ERROR: The array length of initial_guess_param should be the same with the integration dimention", file=sys.stderr)
            exit()
        for i in np.arange(Ndim):
            a = range[i][0]
            b = range[i][1]
            N = Ngrid[i]
            key = int(i)

            mean = initial_guess_param[i][0]
            std =  initial_guess_param[i][1]
            
            cdf_list = np.linspace(stats.norm.cdf(a, loc=mean, scale=std), stats.norm.cdf(b, loc=mean, scale=std), N+1)
            
            bdr = stats.norm.ppf(cdf_list, loc=mean, scale=std)
            bdr[0], bdr[-1] = a, b

            length = np.diff(bdr)
            pdf_list = 1/length/N
            
            Boundary[key] = bdr
            Gridlen[key] = length
            qx[key] = pdf_list
    elif initial_guess == 'vegas':
        trial_vegas_output = initial_guess_param[0]
        #'boundary', 'qx', 'contribution', 'integration', 'COV', 'Kopt', 'flag', 'sample', 'q_star_sample', 'sample_init', 'q_star_sample_init', 'sample_all', 'q_star_sample_all', 'smplidx', 'Len'])
        Kopt = trial_vegas_output.Kopt
        Boundary = trial_vegas_output.boundary[Kopt]
        Gridlen = trial_vegas_output.Len
        qx = trial_vegas_output.qx[Kopt]

    return Boundary, Gridlen, qx


def Sample_VEGAS_grid(Boundary, Boundary0, qx, qx0, Ngrid, Nsmpl, sampling_method):
    Ndim = len(Ngrid)
    smplqx = np.zeros((Nsmpl, Ndim))
    smplidx = np.zeros((Nsmpl, Ndim)).astype(int)
    smplidx0 = np.zeros((Nsmpl, Ndim)).astype(int)
    smplqx0 = np.zeros((Nsmpl, Ndim))
    smpl = np.zeros((Nsmpl, Ndim))
    
    for i_dim, N in enumerate(Ngrid):
        if sampling_method == 'LHS':
            smplidx[:,i_dim] = np.arange(N)
            np.random.shuffle( smplidx[:,i_dim] )
        elif sampling_method == 'none':
            smplidx[:,i_dim] = np.random.randint(0, N, size=Nsmpl) ## Pick random VEGAS grid
        idx = smplidx[:,i_dim].astype(int)
        
        start_points = Boundary[i_dim][idx]
        end_points = Boundary[i_dim][idx + 1]
        smpl[:,i_dim] = np.random.uniform(start_points, end_points) # Generate random numbers within the grid
        smplqx[:,i_dim] = qx[i_dim][ smplidx[:,i_dim] ] 

        smplidx0[:,i_dim] = np.searchsorted(Boundary0[i_dim], smpl[:,i_dim])-1
        smplqx0[:,i_dim] = qx0[i_dim][ smplidx0[:,i_dim] ]

    return smpl, smplidx, smplqx, smplqx0



'''def Sample_LHS_VEGAS_grid(Boundary, Boundary0, qx, qx0, Ngrid, Nsmpl):
    Ndim = len(Ngrid)
    smplqx = np.zeros((Nsmpl, Ndim))
    smplidx = np.zeros((Nsmpl, Ndim)).astype(int)
    smplidx0 = np.zeros((Nsmpl, Ndim)).astype(int)
    smplqx0 = np.zeros((Nsmpl, Ndim))
    smpl = np.zeros((Nsmpl, Ndim))

    for i_dim, N in enumerate(Ngrid):
        smplidx[:,i_dim] = np.random.shuffle(N)
        #smplidx[:,i_dim] = np.random.randint(0, N, size=Nsmpl) ## Pick random VEGAS grid
        idx = smplidx[:,i_dim].astype(int)
        
        start_points = Boundary[i_dim][idx]
        end_points = Boundary[i_dim][idx + 1]
        print(idx, start_points, end_points)
        print()
        smpl[:,i_dim] = np.random.uniform(start_points, end_points) # Generate random numbers within the grid
        smplqx[:,i_dim] = qx[i_dim][ smplidx[:,i_dim] ] 
    
        smplidx0[:,i_dim] = np.searchsorted(Boundary0[i_dim], smpl[:,i_dim])-1
        smplqx0[:,i_dim] = qx0[i_dim][ smplidx0[:,i_dim] ]
        #print("in sample vegas grid", qx0[i_dim])
        #print("in sample vegas grid", smplqx0[:,i_dim])
        #smplqx0 = qx[i_dim][ smplidx0[:,i_dim] ]
        
    return smpl, smplidx, smplqx, smplqx0'''

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

def calc_integ_CV(fx, Nsmpl, smpl, smplidx, qx, smplqx0, integ_range, integ_hist, CV_hist, kth_iter):
    integ = 0
    Ndim = len(smpl.T)
    ith_smpl_ith_dim_grid = smplidx.astype(int)
    
    integ_range = np.array(integ_range)

    ## compute importance weight ##
    wt_array = np.ones(Nsmpl)
    for i_dim in range(Ndim):
        smplqx = qx[i_dim][ith_smpl_ith_dim_grid[:, i_dim]]
        wt_array *= smplqx0[:,i_dim] / smplqx
        #print(kth_iter, wt_array)
    integ_frac=np.zeros(Nsmpl)

    normalizing_constant = np.prod(smplqx0, axis=1)

    ## compute integration ##
    integ_frac = fx*wt_array / normalizing_constant
    integ = np.mean(integ_frac)

    ## compute COV (CV) ##
    integ_frac_sq = integ_frac * integ_frac
    integ_frac_sq_mean = np.mean(integ_frac_sq)
    VAR = ( integ_frac_sq_mean - integ**2) / Nsmpl
    if integ ==0:
        CV=0
    else:
        CV = VAR**0.5 / np.abs(integ)
    ##################

    integ_hist = np.append(integ_hist, integ)
    CV_hist = np.append(CV_hist, CV)

    return integ_hist, CV_hist
#########################################

def Calc_contribution2(fx, qx, bdr, Len, Ngrid, smpl, smplidx, smplqx, d_hist, d_damp_hist, kth_iter, alpha):
    optimizer_frac = np.zeros_like(smpl)
    fx_sq = np.zeros(len(smpl))
    d = copy.deepcopy(qx)
    d = {key: np.zeros_like(value) for key, value in d.items()}
    d_damp = {key: np.zeros_like(value) for key, value in d.items()}
    
    ## 1. CALCULATING ACTUAL CONTRIBUTION
    fx_sq = fx*fx
    #print(np.c_[smpl, fx_sq, smplqx])
    if np.sum(fx_sq)==0:
        print("ERROR: Too small functional values!", file=sys.stderr)
        d_hist[kth_iter] = copy.deepcopy(d)
        d_damp_hist[kth_iter] = copy.deepcopy(d_damp)
        return d, d_damp, d_hist, d_damp_hist

    for i_dim in d: ## run for dimension
        ####### Compute Optimizer ########
        smplqx_i_ne_j = np.delete(smplqx, i_dim, axis=1)
        smplqx_prod = np.prod(smplqx_i_ne_j, axis=1)
        optimizer_frac[:,i_dim] = fx_sq / smplqx_prod # fractional optimizer for i_dim
        idx_i_dim = smplidx[:,i_dim]
        d[i_dim] = np.bincount(idx_i_dim, weights=optimizer_frac[:, i_dim], minlength=Ngrid[i_dim])
        d[i_dim][:] = np.sqrt(d[i_dim][:])

        d[i_dim]=d[i_dim]*Len[i_dim]
        d[i_dim][:] /=np.sum(d[i_dim][:])
        ##################################

        ####### Smoothing Optimizer ########
        d_damp[i_dim][0] = (7 * d[i_dim][0] + d[i_dim][1]) / 8. ###
        d_damp[i_dim][-1] = (d[i_dim][-2] + 7 * d[i_dim][-1]) / 8. ###
        d_damp[i_dim][1:-1] = (d[i_dim][:-2] + 6 * d[i_dim][1:-1] + d[i_dim][2:]) / 8. ###
        ##################################
        if kth_iter < 100:
            ####### Damping Optimizer ########
            # For elements where d_damp[i_dim] > 0
            mask = d_damp[i_dim] > 0 ###
            A = 1 - d_damp[i_dim][mask] ###
            B = -np.log(d_damp[i_dim][mask]) ###
            logX = alpha * (np.log(A) - np.log(B)) ###
            d_damp[i_dim][mask] = np.exp(logX) ###            
            
            # For elements where d_damp[i_dim] <= 0, set to 0
            d_damp[i_dim][~mask] = 0 ###

            d_damp[i_dim]/=np.sum(d_damp[i_dim])
            ##################################
        else:
            d_damp = copy.deepcopy(d)
    
    d_hist[kth_iter] = copy.deepcopy(d)
    d_damp_hist[kth_iter] = copy.deepcopy(d_damp)

    return d, d_damp, d_hist, d_damp_hist

def Update_grid_vectorized(Boundary, qx, Len, Ngrid, d, Boundary_hist, qx_hist, kth_iter_plus_one):
    K = Ngrid[0]*SUBGRID_MUTIPLIER
    ## 1. CALCULATE # OF SUBGRID PER GRID
    M = {key: np.round(d[key] * K) for key in d} 
    for i_dim in M: ## prevent any number M to be 0
        for i, Mval in enumerate(M[i_dim]):
            if M[i_dim][i] == 0:
                M[i_dim][i]=1
        while K - np.sum(M[i_dim][:-1]) < 0.5:
            M[i_dim]=subtract_one_random_row(M[i_dim]) 
        M[i_dim][-1] = K - np.sum(M[i_dim][:-1])
    
    #############################################################################
    ## 2. SUBDIVIDE AND MERGE BOUNDARY : vectorized
    subgrid_boundaries = {key: [] for key in Boundary}
    subgrid_step = {key: [] for key in Boundary}
    for i_dim in Boundary:
        subgrid_boundaries[i_dim] = np.zeros(K+1)
        subgrid_step[i_dim] = np.diff(Boundary[i_dim]) / M[i_dim]
        steps = np.repeat(subgrid_step[i_dim], M[i_dim].astype(int))
        subgrid_boundaries[i_dim] = np.insert(Boundary[i_dim][0] + np.cumsum(steps), 0, Boundary[i_dim][0])
        #print(subgrid_boundaries[i_dim])
        Boundary[i_dim] = subgrid_boundaries[i_dim][::int(K/Ngrid[i_dim])]
    ############################################################################

    Len = {key: np.diff(Boundary[key]) for key in Boundary}
    for i_dim in Boundary:
        for j in range(Ngrid[i_dim]):
            qx[i_dim][j] = 1 / ( Ngrid[i_dim]*Len[i_dim][j] )

    ### Convert list to numpy array. This is mandatory for the use in Sample_VEGAS
    for key in Boundary:
        Boundary[key] = np.array(Boundary[key])

    Boundary_hist[kth_iter_plus_one] = copy.deepcopy(Boundary)
    qx_hist[kth_iter_plus_one] = copy.deepcopy(qx)

    return Boundary, Len, qx, Boundary_hist, qx_hist
#########################################

def Sensitivity_analysis(sample, Bdr, qx):
    obtain_smpl_q(sample, Bdr, qx)

    return 0

def Resample_VEGAS_grid(Boundary, qx, Ngrid, Nsmpl):
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
    
    return smpl, smplqx


def obtain_smpl_q(sample, Bdr, qx):
    """
    Calculates the value of q given N input values.
    
    inputs: List of N input values, where inputs[i] corresponds to the i-th dimension.
    Bdr: Dictionary of boundaries for each dimension.
    qx: Dictionary of function values for each dimension.
    
    Returns:
    Product of selected qx values.
    """
    Nsmpl = len(sample)  # Determine the dimension N
    Ndim = np.shape(sample)[1]
    Q = np.ones(Nsmpl)
    for i_dim in range(Ndim):
        X = sample[:,i_dim]
        B = Bdr[i_dim] # find indices
        q = qx[i_dim]

        grid_index = np.searchsorted(B, X, side='left')-1
        grid_index[grid_index == len(B)-1 ] = len(B) - 2
        grid_index[grid_index == -1 ] = 0

        Q = Q*q[grid_index]
    return Q


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

def Run_vegas_figure(vegas_output, integ_range, Ngrid, Nsmpl_show=1000, num_bins=100):
    Bdr_hist, qx_hist, integ_hist, CV_hist, sus_flag = vegas_output.boundary, vegas_output.qx, vegas_output.integration, vegas_output.COV, vegas_output.flag
    sample_all = vegas_output.sample_all
    q_star_sample_all = vegas_output.q_star_sample_all

    colors = np.array([ '#b0afaf','#5da2b7','#337ca0','#22638e','#304971','#384366','#0b1016', # bluish
               '#f7ffd4', '#d6e7b2', '#5fa941', '#00824c',  #greenish
               '#fadee9', '#fdafc8', '#f03773', '#af2359', # redish
               ])
    #colors = ['#0000ff', '#0000e6', '#0000cc', '#0000b3', '#000099', '#000080', '#000066', '#00004d', '#000033', '#00001a', '#ff0000', '#e60000', '#cc0000', '#b30000', '#990000', '#800000', '#660000', '#4d0000', '#330000', '#1a0000', '#00ff00', '#00e600', '#00cc00', '#00b300', '#009900', '#008000', '#006600', '#004d00', '#003300', '#001a00']

    Niter=len(integ_hist)
    Ndim = len(integ_range)
    
    ## Displaying VEGAS GRID in 2D ##
    if Ndim ==2:
        fig1, ax1 = plt.subplots( figsize=(4,8) )
        fig1.suptitle("VEGAS grid")
        i=Niter-1
        
        x1_min = integ_range[0,0]
        x1_max = integ_range[0,1]
        x2_min = integ_range[1,0]
        x2_max = integ_range[1,1]

        Bdr = Bdr_hist[i]
        qx = qx_hist[i]

        smpl_show, smpl_show_qx = Resample_VEGAS_grid(Bdr, qx, Ngrid, Nsmpl_show)

        for bdr in Bdr_hist[i][0]:
            ax1.plot([bdr,bdr],[x2_min, x2_max], color='black')
        for bdr in Bdr_hist[i][1]:
            ax1.plot([x1_min, x1_max], [bdr,bdr], color='black')
        ax1.scatter(smpl_show[:,0], smpl_show[:,1], color='gray', s=1)
    ######################################


    ### Obtaining marginal distribution of the optimal density ###
    marginal_pdfs_optimal2, bin_centers_optimal2 = marginalize_pdf (sample_all, q_star_sample_all, integ_range, num_bins)

    ## Displaying Optimal Density ##
    if Ndim ==2:
        fig_optimal_all, ax_optimal_all = plt.subplots()
        scatter=ax_optimal_all.scatter(sample_all[:,0],sample_all[:,1], c=q_star_sample_all, s=1, cmap='rainbow')
        colorbar=plt.colorbar(scatter)
        ax_optimal_all.set_title("Optimal density")
        #fig_optimal.colorbar(ax_optimal.contourf(grid_x, grid_y, z, cmap='rainbow'))
    #########################################

    ## Displaying marginal distributions ###
    fig, ax = plt.subplots( Ndim, figsize=(4,8))
    if len(integ_range) == 1:
        ax = [ax]
    fig.suptitle("Deaggregation")
    for i in range(Niter): # ith iteration
        color_i = i
        COVlabel = "COV=" + f'{CV_hist[i]*100:.2f}' + "%"
        
        for i_dim in range(0, Ndim): # i_dimth dimention
            # print(i_dim, len(integ_range))
            x_values = np.ravel(np.column_stack((Bdr_hist[i][i_dim][:-1], Bdr_hist[i][i_dim][1:]))).tolist() # make a new array for plotting using plt.plot
            y_values = np.ravel(np.column_stack((qx_hist[i][i_dim], qx_hist[i][i_dim]))).tolist() # make a new array for plotting using plt.plot
            plot_arr = np.c_[x_values, y_values] # make a new array for plotting using plt.plot
            if i< len(colors):
                ax[i_dim].plot(plot_arr[:,0], plot_arr[:,1], color=colors[color_i], label=COVlabel)
            else:
                ax[i_dim].plot(plot_arr[:,0], plot_arr[:,1], label=COVlabel)

        #ax[0].legend()
    
    #### plot optimal density
    for i_dim in range(0, Ndim): # i_dimth dimention
        ax[i_dim].plot(bin_centers_optimal2[:,i_dim], marginal_pdfs_optimal2[:,i_dim], label="optimal", color = 'red')

    fig.tight_layout()

    fig_cov, ax_cov = plt.subplots()
    ax_cov.plot(range(Niter), CV_hist*100, color='black', alpha=0.5)
    for i in range(Niter): # ith iteration
        color_i = i
        COVlabel = "COV=" + f'{CV_hist[i]*100:.2f}' + "%"
        if i < len(colors):
            ax_cov.scatter(i, CV_hist[i]*100, color=colors[color_i], label=COVlabel)
        else:
            ax_cov.scatter(i, CV_hist[i]*100, label=COVlabel)
    ax_cov.set_xlabel("Iteration")
    ax_cov.set_ylabel("COV (%)")
    ax_cov.set_yscale("log")
    fig_cov.legend()
    fig_cov.tight_layout()



    plt.show()
    #####################################

    return fig, ax
    
def marginalize_pdf(sample, pdf_values, integ_range, num_bins, dim_list=[]):
    """
    By default, output 1-d marginalized pdfs.
    if dim_list is specified the pdf are marginlaized to those dimensions.
        e.g., if dim_list = [0, 2], we obtain two-dimensional pdf in first(0) and third(2) sample dimension
    """
    if dim_list == []:
        # 1. Obtain evenly spaced, interpolated joint distribution
        # Generate a regular grid in N-dimensional space
        bounds = [(np.min(integ_range[i, 0]), np.max(integ_range[i, 1])) for i in range(sample.shape[1])]
        dxs = (integ_range[:,1] - integ_range[:,0]) / num_bins
        Ndim = len(integ_range)
        Nsmpl = len(sample)
        marginal_pdfs_optimal, bin_centers_optimal = np.zeros((num_bins, Ndim)), np.zeros((num_bins, Ndim))
        #print(dx)
        #grid = np.meshgrid(*[np.linspace(bound[0], bound[1], num_bins) for bound in bounds])
        #print(grid)
        #grid_points = np.vstack([g.ravel() for g in grid]).T  # NxK array where K is the number of grid points
        for dim in range(Ndim):
            bound, dx = bounds[dim], dxs[dim]
            range_dim = bound[1]-bound[0]
            for j in range(num_bins):
                x1 = bound[0] + j*dx
                x2 = bound[0] + (j+1)*dx
                x_cen = (x1+x2)/2
                indices = np.where((sample[:, dim] > x1) & (sample[:, dim] < x2))[0]
                marginalized_pdf = np.mean(pdf_values[indices])
                bin_centers_optimal[j,dim] = x_cen
                marginal_pdfs_optimal[j,dim] = marginalized_pdf
            marginal_pdfs_optimal[:,dim] /= np.sum(marginal_pdfs_optimal[:,dim])
            marginal_pdfs_optimal[:,dim] /= dx

        return marginal_pdfs_optimal, bin_centers_optimal
    else:
        ### SET GRID CENTER COORDINATES ### 
        bounds = np.array([ (np.min(integ_range[i, 0]), np.max(integ_range[i, 1])) for i in dim_list ])
        dxs = np.array([ ( (integ_range[i,1] - integ_range[i,0]) / num_bins ) for i in dim_list ])
        full_edges = np.array([np.arange(bounds[i, 0], bounds[i, 1] + dxs[i]/2, dxs[i]) for i in range(len(bounds))])
        midpoints = full_edges[:, :-1] + dxs[:, None] / 2

        grid_combinations = np.meshgrid(*midpoints, indexing='ij')
        grid_centers_optimal = np.vstack([grid.ravel() for grid in grid_combinations]).T
        #####################################


        num_grids = len(grid_centers_optimal)
        num_sample = len(sample)
        num_dim = len(dim_list)
        
        x1_list = (grid_centers_optimal - dxs/2)
        x2_list = (grid_centers_optimal + dxs/2)
        
        x1_expanded = x1_list[np.newaxis, :, :]  # Shape: (1, num_din, nun_grids)
        x2_expanded = x2_list[np.newaxis, :, :]  # Shape: (1, num_din, nun_grids)
        
        # Expand sample and x1_list, x2_list for broadcasting
        sample_selected_dim = sample[:,dim_list]
        sample_expanded = sample_selected_dim[:, np.newaxis, :]  # Shape: (Nsample, 1, Ndim)
        x1_expanded = x1_list[np.newaxis, :, :]  # Shape: (1, Ngrid, Ndim)
        x2_expanded = x2_list[np.newaxis, :, :]  # Shape: (1, Ngrid, Ndim)

        # Check if each sample is within the respective bounds across all dimensions
        in_bounds = (sample_expanded >= x1_expanded) & (sample_expanded <= x2_expanded)  # Shape: (Nsample, Ngrid, Ndim)

        # Determine if all dimension conditions are met for each grid
        all_conditions_met = np.all(in_bounds, axis=2)  # Shape: (Nsample, Ngrid)

        # Find the first grid index where conditions are met for each sample
        index = np.argmax(all_conditions_met, axis=1)  # Get the first index of True along the Ngrid dimension


        # Handle cases where no grid matches by checking if any condition was True
        no_match = ~np.any(all_conditions_met, axis=1)  # True if no conditions met
        index[no_match] = -1  # Assign -1 or any identifier for no match

        # Compute the sum of pdf_values for each unique index
        sum_pdf = np.bincount(index, weights=pdf_values, minlength=num_grids)

        # Compute the count of occurrences for each unique index
        count_pdf = np.bincount(index, minlength=num_grids)

        # Calculate the average by dividing sum by count
        # Use np.where to handle division by zero gracefully
        marginalized_pdf = np.where(count_pdf > 0, sum_pdf / count_pdf, 0)
        marginalized_pdf /= np.sum(marginalized_pdf)
        volume = np.prod(x2_list - x1_list, axis=1)
        
        # Normalize to make the area = 1
        marginal_pdfs_optimal = marginalized_pdf / volume
        
        return marginal_pdfs_optimal, grid_centers_optimal
    




## For the use in later ##
'''def get_qx_from_bdr(Boundary, Ngrid):
    qx = {}
    Ndim = len(list(Boundary.keys()))
    for i in np.arange(Ndim):
        N = Ngrid[i]
        key = int(i)
        qx[key] = np.zeros( N ) 

    Len = {key: np.diff(Boundary[key]) for key in Boundary}
    for i_dim in Boundary:
        for j in range(Ngrid[i_dim]):
            qx[i_dim][j] = 1 / ( Ngrid[i_dim]*Len[i_dim][j] )
    return qx'''

'''def smoothing_qx(bdr, qx, Ngrid, integ_range):
    Nx = 100000
    Ndim = len(Ngrid)
    x = np.zeros((Nx, Ndim))
    x_qx_smooth = np.zeros((Nx, Ndim))

    for i_dim in bdr:

        ## interpolation
        qx_left = np.insert(qx[i_dim], len(qx[i_dim]), qx[i_dim][-1])
        qx_right = np.insert(qx[i_dim], 0, [qx[i_dim][0]])
        qx_avg = (qx_left + qx_right) /2
        
        cs = PchipInterpolator(bdr[i_dim], qx_avg)
        x0 = integ_range[i_dim,0]
        x1 = integ_range[i_dim,1]
        dx = ((x1-x0)/Nx) 
        
        x[:,i_dim] = np.linspace(x0, x1, Nx)
        qx_interp = cs(x[:,i_dim])
        qx_interp /= np.sum( qx_interp * dx )
        

        ## smoothing
        smoothing_kernel = np.ones( int(Nx/20) ) / (Nx/20.0)

        qx_interp_for_smooth = np.insert(qx_interp, 0, qx_interp[:int(len(smoothing_kernel)/2-1)])
        qx_interp_for_smooth = np.insert(qx_interp_for_smooth, len(qx_interp_for_smooth), qx_interp[-int(len(smoothing_kernel)/2):])
        
        qx_interp_smooth = np.convolve(qx_interp_for_smooth, smoothing_kernel , mode='valid')
        
        
        qx_interp_smooth /= np.sum( qx_interp_smooth * dx ) ## normalization



        x_qx_smooth[:,i_dim] = qx_interp # return interpolated version
        #x_qx_smooth[:,i_dim] = qx_interp_smooth # returm interpolated + moving average applied version
    return x, x_qx_smooth'''

###################################################



'''def Update_grid(Boundary, qx, Len, Ngrid, d, Boundary_hist, qx_hist, kth_iter_plus_one):
    K = Ngrid[0]*1_000
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

    Boundary_hist[kth_iter_plus_one] = copy.deepcopy(Boundary)
    qx_hist[kth_iter_plus_one] = copy.deepcopy(qx)

    return Boundary, Len, qx, Boundary_hist, qx_hist
#########################################'''