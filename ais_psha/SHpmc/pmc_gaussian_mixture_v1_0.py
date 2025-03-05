import numpy as np
from numpy.linalg import cholesky, inv
from scipy.stats import norm
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import sys 
from collections import namedtuple
from matplotlib.patches import Ellipse

PMCoutput = namedtuple('PMCoutput', ['integ', 'COV', 'integ_list', 'COV_list', 'gmModel'
                                     , 'sample_list', 'q_star_list', 'sample_final', 'q_star_final'
                                     , 'Niter'])

from sklearn.mixture import GaussianMixture

# version 0_1
# add output sample and optimal density

def Run_pmc_normal_integ3(f, integ_range, Nsmpl=100, Niter=20, KS_statistic_threshold=0.1, display=False, init_gmModel = '', init_pmc_output='', Ngaussian=1, show_integ=True, correlation=True, **f_kwargs):
    """
    f : integrand (target distribution)
    integ_range : integration range
    Nsmpl : number of samples
    Niter : Maximum number of iterations
    KS_statistic_threshold: stopping criteria
    Ngaussian : The number of gaussian mixture model
    """

    # Initialize #
    Ndim, ks_stat, X_i, qX_i = initialize2(integ_range, Nsmpl, init_gmModel)
    gmModel = GaussianMixture(n_components=Ngaussian)
    sample_list, fX_list, wt_list, q_star_list = np.empty((Nsmpl, Niter, Ndim)), np.empty((Nsmpl, Niter)), np.empty((Nsmpl, Niter)), np.empty((Nsmpl, Niter))
    integ_list, stdev_list, COV_list = np.empty(Niter), np.empty(Niter), np.empty(Niter)
    ##############
    
    for i in range(Niter):
        
        ### FOR THIS LOOP ###

        ## Calc functional value ##
        fX_i = f(X_i, **f_kwargs)
        fX_i = np.array(fX_i).ravel() ### Make sure that the output is 1-d array
        if len(np.unique(fX_i)) <2:
            raise ValueError(f"The model esitmate has no variability. All estimates are {np.unique(fX_i)}. Possibly due to bad initial guess.")

        ## update sample pdf ##
        wt_i = fX_i/qX_i
        wt_i_abs = np.fabs(wt_i)
        
        ## calc integration ##
        integ_i = np.mean(wt_i)

        var_i = (np.mean(wt_i*wt_i) - integ_i**2) / Nsmpl
        stdev_i = np.sqrt( var_i )
        COV_i = stdev_i/integ_i
        
        integ_list[i] = integ_i
        stdev_list[i] = stdev_i
        COV_list[i] = COV_i
        

        ## Stack all the sample and functional values ##
        sample_list[:,i,:] = X_i
        fX_list[:,i] = fX_i
        q_star_list = fX_list / integ_i
        q_star_i = fX_i/integ_i
        wt_list[:,i] = wt_i


        ### FOR NEXT LOOP ###

        ## Re-select X (X2) based on wt. If propose density = target density, p = uniform distribution,
        ## i.e., X2 is sampled from exactly same distribution with X, converge
        X_j = X_i[ np.random.choice( len(X_i), size=len(X_i), p=wt_i_abs/np.sum(wt_i_abs) ) ]
        
        # Update Gaussian Mixture Model
        gmModel.fit(X_j)
        
        ### Replace X_j with the one directly sampled from updated distribution,
        ### to prevent X_j to have dupliates
        X_j, _ = gmModel.sample(n_samples=Nsmpl)

        ## update sample pdf ##
        qX_j = np.exp(gmModel.score_samples(X_j))
        

        ## Stopping criteria: if sample pdf does not change, STOP ##
        for i_dim in range(Ndim):
            ks_stat[i_dim] = ks_2samp(X_i[:,i_dim], X_j[:,i_dim])[0]
        if np.all(ks_stat < KS_statistic_threshold):
            #integ_wt, COV_wt = weighted_average_integ(integ_list, stdev_list, COV_list)
            
            print_output(i, integ_i, COV_i, display, Ndim, integ_range, X_i, X_j, gmModel, show_integ)
            #sample_list, q_star_list = filterout_samples_out_of_range(integ_range, sample_list, q_star_list)# filter out samples out of the integration range
            #sample, q_star = filterout_samples_out_of_range(integ_range, X_i, q_star)
            return PMCoutput(integ_i, COV_i, integ_list, COV_list, gmModel, sample_list, q_star_list, X_i, q_star_i, i)

        ## replace ##
        X_i, qX_i = X_j, qX_j

    #integ_wt, COV_wt = weighted_average_integ(integ_list, stdev_list, COV_list)
    print_output(i, integ_i, COV_i, display, Ndim, integ_range, X_i, X_j, gmModel, show_integ)
    #sample_list, q_star_list = filterout_samples_out_of_range(integ_range, sample_list, q_star_list)
    #sample, q_star = filterout_samples_out_of_range(integ_range, X_i, q_star)
    
    return PMCoutput(integ_i, COV_i, integ_list, COV_list, gmModel, sample_list, q_star_list, X_i, q_star_i, i)


def weighted_average_integ(integ_list, stdev_list, COV_list):
    #idx = np.where(np.array(COV_list)<0.1,1,0)
    integ_list = np.array(integ_list)
    stdev_list = np.array(stdev_list)
    COV_list = np.array(COV_list)

    COV_min = np.min(COV_list)
    COV_cri = COV_min*2

    integ_list=integ_list[COV_list<COV_cri]
    stdev_list=stdev_list[COV_list<COV_cri]

    var_list = stdev_list*stdev_list
    wt_list = 1/var_list
    var = 1/np.sum(wt_list)
    integ = np.sum(integ_list*wt_list) / np.sum(wt_list)
    COV = var**0.5/integ
    return integ, COV

def filterout_samples_out_of_range(integ_range, sample, q_star):
    sample = sample.transpose(1,0,2).reshape(-1, len(integ_range))
    q_star = q_star.T.flatten()
    pnp = np.zeros_like(sample)
    for i,range in enumerate(integ_range):
        pnp[:,i] = np.where( (sample[:,i]>range[0]) & (sample[:,i]<range[1]), 1, 0)
    pnp = np.prod(pnp, axis=1)
    sample_filtered = sample[pnp==1]
    q_star_filtered = q_star[pnp==1]
    return sample_filtered, q_star_filtered




def initialize2(integ_range, Nsmpl, init_gmModel):
    integ_range = np.array(integ_range)
    
    # Check whether this is 1-d problem
    if integ_range.ndim ==1:
        integ_range = integ_range[np.newaxis,:]
    
    Ndim = len(integ_range)
    
    ks_stat = np.full(Ndim, 10.0)
    if init_gmModel=='':
        X = np.random.uniform(integ_range[:,0], integ_range[:,1], size=(Nsmpl, Ndim))
        qX = np.full(len(X), np.prod(integ_range[:,1]-integ_range[:,0]))
        
    else:
        X, _ = init_gmModel.sample(n_samples=Nsmpl)
        qX = np.exp(init_gmModel.score_samples(X))
    #print(qX)

    return Ndim, ks_stat, X, qX

def set_gmm(gmm, mean, wt, cov):
    gmm.means_=mean
    gmm.weights_=wt
    gmm.covariances_=cov
    precisions_cholesky = []
    for cov_i in cov:
        cholesky_decomp = cholesky(inv(cov_i))  # Compute the Cholesky decomposition of the precision matrix
        precisions_cholesky.append(cholesky_decomp)
    gmm.precisions_cholesky_ = np.array(precisions_cholesky)
    return gmm


def print_output(i, integ, COV, display, Ndim, integ_range, X, X_new, gmModel_new, show_integ):
    ## show the result as a prompt ##
    if show_integ ==True:
        print("converged at i = ", i, ","
            , "Integ = ", "{:.12e}".format(integ)
            ,  "COV = ", "{:.2f}".format(COV*100), "%"
            , file=sys.stderr)
    
    ## show the result figure ##
    if display == True:
        fig, ax = plt.subplots(Ndim)
        for i in range(Ndim):
            ax[i].set_xlim(integ_range[i,:])
            ax[i].hist(X[:,i], label="old", density=True, alpha=1.0, bins=20, color='gray')
            ax[i].hist(X_new[:,i], label="new", density=True, alpha=0.5, bins=20, color='orange')
            ax[i].plot(np.sort(X_new[:,i]), norm.pdf(np.sort(X_new[:,i]), loc=gmModel_new.means_[i], scale=gmModel_new.covariances_[i,i]**0.5), color='orange')
            ax[i].legend()
        fig.tight_layout()
        plt.show()
    ###
 

def Run_pmc_figure(pmc_output, integ_range, num_bins=50, method='nearest'):
    Ndim = len(integ_range)
    #mean = pmc_output.gmModel.means_
    #cov = pmc_output.gmModel.covariances_
    sample = pmc_output.sample_list.transpose(1,0,2).reshape(-1, Ndim)
    q_star = pmc_output.q_star_list.T.flatten()
    print(np.shape(sample))
    print(np.shape(q_star))

    if Ndim ==2:
        pdfminmax = plot_gmm_contour_from_sklearn(pmc_output.gmModel, integ_range, resolution=100)
        plot_point_estimate_optimal(sample, q_star, integ_range, vminmax=pdfminmax)
        plot_LSF(sample, q_star, integ_range)
    
    marginal_pdfs_optimal, bin_centers_optimal = marginalize_pdf(sample, q_star, integ_range, num_bins)
    
    plot_marginal(Ndim, marginal_pdfs_optimal, bin_centers_optimal)

def plot_marginal(Ndim, marginal_pdfs_optimal, bin_centers_optimal):
    #### plot optimal density
    fig, ax = plt.subplots(Ndim)
    for i_dim in range(0, Ndim): # i_dimth dimention
        x_plot = bin_centers_optimal[:,i_dim]
        x_interval = x_plot[1]-x_plot[0]
        if i_dim == 0:
            ax[i_dim].bar(x_plot, marginal_pdfs_optimal[:,i_dim], width=x_interval, label="optimal", color = 'red')
            #ax[i_dim].plot(x_plot, norm.pdf(x_plot,loc=mean[i_dim], scale=cov[i_dim,i_dim]**0.5), label="fitted", color = 'black')
        else:
            ax[i_dim].bar(x_plot, marginal_pdfs_optimal[:,i_dim], width=x_interval, color = 'red')
            #ax[i_dim].plot(x_plot, norm.pdf(x_plot,loc=mean[i_dim], scale=cov[i_dim,i_dim]**0.5), color = 'black')
    fig.legend()
    fig.tight_layout()


def plot_point_estimate_optimal(sample, q_star, integ_range, vminmax):
    fig_optimal_all, ax_optimal_all = plt.subplots(figsize=(8,6))
    scatter=ax_optimal_all.scatter(sample[:,0],sample[:,1], c=q_star, s=1, cmap='rainbow', vmin = vminmax[0], vmax = vminmax[1])
    ax_optimal_all.set_title("Optimal density")
    ax_optimal_all.set_xlim(integ_range[0,0], integ_range[0,1])
    ax_optimal_all.set_ylim(integ_range[1,0], integ_range[1,1])
    fig_optimal_all.colorbar(scatter, label="pdf")
    fig_optimal_all.tight_layout()

def plot_LSF(sample, q_star, integ_range):
    fig, ax = plt.subplots(figsize=(8,6))
    LSF = np.where(q_star>10e-300, 1, 0)
    ax.scatter(sample[LSF==1,0],sample[LSF==1,1], s=1, c='red', label = "exceeded")
    ax.scatter(sample[LSF==0,0],sample[LSF==0,1], s=1, c='blue', label = "NOT exceeded")
    ax.set_title("Limit State")
    ax.set_xlim(integ_range[0,0], integ_range[0,1])
    ax.set_ylim(integ_range[1,0], integ_range[1,1])
    ax.legend()
    ax.grid(alpha=0.5)
    fig.tight_layout()
    #fig.colorbar(scatter, label="pdf")


def plot_gmm_contour_from_sklearn(gmm, integ_range, resolution=100, ax=None):
    """
    Plots a Gaussian Mixture Model (GMM) as a contour plot in 2D space
    using Scikit-learn's GaussianMixture output.
    
    Parameters:
        gmm (GaussianMixture): Trained Gaussian Mixture model.
        X (np.ndarray): Data points (used to set axis limits).
        resolution (int): Number of grid points for each axis.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))

    # Define the grid over which to evaluate
    #ax_optimal_all.set_title("Optimal density")
    #ax_optimal_all.set_xlim(integ_range[0,0], integ_range[0,1])
    #ax_optimal_all.set_ylim(integ_range[1,0], integ_range[1,1])

    x_min, x_max = integ_range[0,0] - 1, integ_range[0,1] + 1
    y_min, y_max = integ_range[1,0] - 1, integ_range[1,1] + 1
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X_grid, Y_grid = np.meshgrid(x, y)
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

    # Compute log-likelihood for each grid point
    log_likelihood = gmm.score_samples(grid_points)
    Z = log_likelihood.reshape(X_grid.shape)
    z = np.exp(Z)

    # Plot the contour
    contour = plt.contourf(X_grid, Y_grid, z, levels=100, cmap='rainbow', vmin=np.min(z), vmax=np.max(z))

    for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        if gmm.covariance_type == 'full':
            cov = covar
        elif gmm.covariance_type == 'tied':
            cov = gmm.covariances_
        elif gmm.covariance_type == 'diag':
            cov = np.diag(covar)
        elif gmm.covariance_type == 'spherical':
            cov = np.eye(len(mean)) * covar
        
        # Calculate the ellipse parameters
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)  # 2 std deviations
        
        # Add the ellipse to the plot
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='black', facecolor='none')
        ax.add_patch(ellipse)

    # Plot the Gaussian means
    ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], color="black", s=10, marker="x")
    ax.set_xlim(integ_range[0,0], integ_range[0,1])
    ax.set_ylim(integ_range[1,0], integ_range[1,1])

    ax.set_title("Gaussian Mixture Model")
    fig.colorbar(contour, label="pdf")

    return [np.min(z), np.max(z)]



def marginalize_pdf(sample, pdf_values, integ_range, num_bins):
    # 1. Obtain evenly spaced, interpolated joint distribution
    # Generate a regular grid in N-dimensional space
    bounds = [(np.min(integ_range[i, 0]), np.max(integ_range[i, 1])) for i in range(sample.shape[1])]
    dxs = (integ_range[:,1] - integ_range[:,0]) / np.array(num_bins)
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
            if len(indices) == 0:
                marginalized_pdf=0
            else:
                marginalized_pdf = np.mean(pdf_values[indices])
            bin_centers_optimal[j,dim] = x_cen
            marginal_pdfs_optimal[j,dim] = marginalized_pdf
        marginal_pdfs_optimal[:,dim] /= np.sum(marginal_pdfs_optimal[:,dim])
        marginal_pdfs_optimal[:,dim] /= dx

    #print(marginal_pdfs_optimal)
    #print(num_bins, np.var(marginal_pdfs_optimal, axis=0))
    return marginal_pdfs_optimal, bin_centers_optimal



def marginalize_pdf_ndim(sample, pdf_values, integ_range, num_bins=100, axis=[1,3]):
    """
    num_bins : array-like. If number, that applies to all dimensions, if array, each element corresponds to the each dimension
    axis : dimensions to be marginalized. e.g., [0,1] means the marginalized pdf values are corresponding to the first and second variables.
    """

    bdr_list, dxs= construct_bdr_list(integ_range, num_bins, axis)
    cubes = construct_small_cubes(bdr_list)

    x = np.zeros((len(cubes), len(axis)))

    marginalized_pdf = np.zeros(len(cubes))
    for i, cube in enumerate(cubes):
        #print(cube)
        for i_dim, ax in enumerate(axis):
            
            #print(i_dim)
            x1 = cube[i_dim*2]
            x2 = cube[i_dim*2+1]
            x[i,i_dim] = (x1+x2)/2

            indice = np.where((sample[:, ax] > x1) & (sample[:, ax] < x2))
            
            if len(indice) == 0:
                indice_common = []
                break

            if i_dim ==0:
                indice_common = indice
            else:
                indice_common = np.intersect1d(indice, indice_prv)
            indice_prv = indice

        if len(indice_common) == 0: 
        #if indice_common[0].size == 0:
            marginalized_pdf[i] = 0
        else:  
            #print(len(indice_common), indice_common, pdf_values[indice_common])
            marginalized_pdf[i] = np.mean(pdf_values[indice_common])

    marginalized_pdf /= np.sum(marginalized_pdf)
    marginalized_pdf /= np.prod(dxs[axis])

    return x, marginalized_pdf


def marginalize_pdf_ndim_AIS(sample, pdf_values, integ_range, axis):
    Ndim = len(integ_range)
    axis_integ =  np.setdiff1d(range(Ndim), axis)
    axis_integ_range = integ_range[axis_integ]
    #print(axis_integ, axis_integ_range)
    #for smpl in sample:
    #    print(smpl)


        
from itertools import product
def construct_bdr_list(integ_range, num_bins, axis):
    dxs = (integ_range[:,1] - integ_range[:,0]) / num_bins
    bdr_list = {}
    cen_list = {}
    for i_dim in axis:
        x0=integ_range[i_dim,0]
        x1=integ_range[i_dim,1]
        dx=dxs[i_dim]
        bdr_list[str(i_dim)] = np.arange(x0, x1+dx/2, dx)
        cen_list[str(i_dim)] = ( bdr_list[str(i_dim)][:-1] + bdr_list[str(i_dim)][1:] ) / 2
    return bdr_list, dxs

def construct_small_cubes(boundaries):
    """
    Construct small cubes from boundary values of each dimension.

    Parameters:
        boundaries (dict): Dictionary where keys are dimensions (0, 1, ..., N-1) 
                           and values are 1D arrays of boundary values.

    Returns:
        np.ndarray: Array of shape (num_cubes, 2 * N), where each row contains
                    the lower and upper bounds of a cube for all dimensions.
    """
    # Sort boundaries for each dimension to ensure order
    sorted_boundaries = {dim: np.sort(boundaries[dim]) for dim in boundaries}
    
    # Find the number of dimensions
    dimensions = list(sorted_boundaries.keys())

    # Generate all intervals for each dimension
    intervals_per_dim = [
        [(sorted_boundaries[dim][i], sorted_boundaries[dim][i + 1])
         for i in range(len(sorted_boundaries[dim]) - 1)]
        for dim in dimensions
    ]

    # Generate all combinations of intervals across dimensions
    cubes = list(product(*intervals_per_dim))
    
    # Flatten the tuples to create an array of shape (num_cubes, 2 * N)
    cubes_array = np.array([np.concatenate(cube) for cube in cubes])

    return cubes_array



def Run_pmc_UQ(out, func_fx, integ_range, axis, num_bins=10, show_figure=False):
    """
    out = PMC output
    fx = out.sample's original pdf function
    integ_range = integration range in 2d array
    axis = axis for UQ
    num_bins = not too small and large
    """

    X, q = marginalize_pdf_ndim(out.sample, out.q_star, integ_range, num_bins=num_bins, axis=axis) ##

    f = func_fx(X, marginalize_axis=axis)

    lam = out.integ

    # sigma for lambda
    wt = np.where(f==0, 0, q/f)
    sigma = np.std(lam*wt)

    cov = sigma/lam

    #V = np.log(1+sigma**2/lam**2)
    
    #lnsigma = np.sqrt(V)

    #lnvar = np.var( np.log(q/f) )
    #lnsigma2 = lnvar**0.5
    #print(lnsigma, lnsigma2)



    ## contour plot
    if len(axis)==2 and show_figure==True:
        from scipy.interpolate import griddata

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Create a grid for contour plotting
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        x_grid, y_grid = np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)



        # Interpolate f_plot and q_plot onto the grid
        f_grid = np.clip(griddata(X, f, (X_grid, Y_grid), method='cubic'), 0,np.inf)
        q_grid = np.clip(griddata(X, q, (X_grid, Y_grid), method='cubic'), 0,np.inf)
        vmax = max(f_grid.max(), q_grid.max())
        contour0 = ax[0].contourf(X_grid, Y_grid, f_grid, cmap='rainbow', vmin=0, vmax=vmax)
        #fig.colorbar(contour0, ax=ax[0])

        # Plot contours for q_plot
        contour1 = ax[1].contourf(X_grid, Y_grid, q_grid, cmap='rainbow', vmin=0, vmax=vmax)
        #fig.colorbar(contour1, ax=ax[1])
        plt.show()

    return sigma, cov



'''def Run_pmc_normal_integ(f, integ_range, Nsmpl, Niter=20, KS_statistic_threshold=0.1, display=False, init_gmModel = '', Ngaussian=5, correlation=True, **f_kwargs):
    """
    f : integrand
    integ_range : integration range
    Nsmpl : number of samples
    Niter : Maximum number of iterations
    KS_statistic_threshold: stopping criteria
    Ngaussian : The number of gaussian mixture model
    """

    # Initialize #
    Ndim, ks_stat, gmModel, X, qX = initialize(integ_range, init_gmModel, Nsmpl, Ngaussian=Ngaussian, correlation=correlation)
    sample_all = np.empty((0, X.shape[1]))
    fX_all = np.empty(0)
    integ_list, stdev_list, COV_list = [],[],[]
    ##############
    
    for i in range(Niter):
        # Calc functional value
        #fX = f(X, **f_kwargs)
        fX = f(X)
        
        ## update sample pdf ##
        wt = fX/qX
        new_pdf =  wt/np.sum(wt)

        ## update parameters ##
        X_new = X[ np.random.choice(len(X), size=len(X), p=new_pdf) ]
        gmModel.fit(X_new)
        
        gmmean_new = gmModel.means_
        gmcov_new = gmModel.covariances_
        gmwt_new = gmModel.weights_

        ## update pdf ##
        gmModel_new = GaussianMixture(n_components=Ngaussian, covariance_type='full',)
        gmModel_new = set_gmm(gmModel_new, gmmean_new, gmwt_new, gmcov_new)

        ## resample from updated pdf ##
        X_new, _ = gmModel_new.sample(n_samples=Nsmpl)

        ## update sample pdf ##
        qX_new = np.exp(gmModel_new.score_samples(X_new))

        ## calc integration ##
        integ = np.mean(wt)
        stdev = np.sqrt( (np.mean(wt*wt) - integ**2) /Nsmpl )
        COV = stdev/integ
        integ_list.append(integ)
        stdev_list.append(stdev)
        COV_list.append(COV)
        
        ## Stack all the sample and functional values ##
        sample_all = np.vstack((sample_all, X))
        fX_all = np.hstack((fX_all, fX))
        q_star_all = fX_all / integ
        q_star = fX/integ

        ## stopping criteria: if sample pdf does not change, STOP ##
        for i_dim in range(Ndim):
            ks_stat[i_dim] = ks_2samp(X[:,i_dim], X_new[:,i_dim])[0]
        if np.all(ks_stat < KS_statistic_threshold):
            integ_wt, COV = weighted_average_integ(integ_list, stdev_list, COV_list)
            print_output(i, integ, COV, display, Ndim, integ_range, X, X_new, gmModel_new)
            sample_all, q_star_all = filterout_samples_out_of_range(integ_range, sample_all, q_star_all)# filter out samples out of the integration range
            sample, q_star = filterout_samples_out_of_range(integ_range, X_new, q_star)
            return PMCoutput(integ, COV, integ_list, COV_list, gmModel_new, sample_all, q_star_all, sample, q_star)

        ## replace ##
        X, qX, gmModel = X_new, qX_new, gmModel_new

    integ_wt, COV = weighted_average_integ(integ_list, stdev_list, COV_list)
    print_output(i, integ, COV, display, Ndim, integ_range, X, X_new, gmModel_new)
    sample_all, q_star_all = filterout_samples_out_of_range(integ_range, sample_all, q_star_all)
    sample, q_star = filterout_samples_out_of_range(integ_range, X_new, q_star)
    return PMCoutput(integ, COV, integ_list, COV_list, gmModel_new, sample_all, q_star_all, sample, q_star)



def Run_pmc_normal_integ2(f, integ_range, Nsmpl, Niter=20, KS_statistic_threshold=0.1, display=False, init_gmModel = '', Ngaussian=5, correlation=True, **f_kwargs):
    """
    f : integrand (target distribution)
    integ_range : integration range
    Nsmpl : number of samples
    Niter : Maximum number of iterations
    KS_statistic_threshold: stopping criteria
    Ngaussian : The number of gaussian mixture model
    """

    # Initialize #
    Ndim, ks_stat, gmModel, X_i, qX_i = initialize(integ_range, init_gmModel, Nsmpl, Ngaussian=Ngaussian, correlation=correlation)
    sample_list, fX_list, wt_list, q_star_list = np.empty((Nsmpl, Niter, Ndim)), np.empty((Nsmpl, Niter)), np.empty((Nsmpl, Niter)), np.empty((Nsmpl, Niter))
    integ_list, stdev_list, COV_list = np.empty(Niter), np.empty(Niter), np.empty(Niter)
    ##############
    #print(X_i)
    for i in range(Niter):
        
        ### FOR THIS LOOP ###

        ## Calc functional value ##
        fX_i = f(X_i)
        #print(np.unique(fX_i)) #############
        
        ## update sample pdf ##
        wt_i = fX_i/qX_i

        ## calc integration ##
        integ_i = np.mean(wt_i)
        stdev_i = np.sqrt( (np.mean(wt_i*wt_i) - integ_i**2) /Nsmpl )
        COV_i = stdev_i/integ_i
        integ_list[i] = integ_i
        stdev_list[i] = stdev_i
        COV_list[i] = COV_i

        ## Stack all the sample and functional values ##
        sample_list[:,i,:] = X_i
        fX_list[:,i] = fX_i
        q_star_list = fX_list / integ_i
        q_star_i = fX_i/integ_i
        wt_list[:,i] = wt_i


        ### FOR NEXT LOOP ###

        ## Re-select X (X2) based on wt. If propose density = target density, p = uniform distribution,
        ## i.e., X2 is sampled from exactly same distribution with X, converge
        X_j = X_i[ np.random.choice( len(X_i), size=len(X_i), p=wt_i/np.sum(wt_i) ) ]
        
        # Update Gaussian Mixture Model
        gmModel.fit(X_j)
        
        ### Replace X_j with the one directly sampled from updated distribution,
        ### to prevent X_j to have dupliates
        X_j, _ = gmModel.sample(n_samples=Nsmpl)

        ## update sample pdf ##
        qX_j = np.exp(gmModel.score_samples(X_j))
        
        

        ## Stopping criteria: if sample pdf does not change, STOP ##
        for i_dim in range(Ndim):
            ks_stat[i_dim] = ks_2samp(X_i[:,i_dim], X_j[:,i_dim])[0]
        if np.all(ks_stat < KS_statistic_threshold):
            #integ_wt, COV_wt = weighted_average_integ(integ_list, stdev_list, COV_list)
            
            print_output(i, integ_i, COV_i, display, Ndim, integ_range, X_i, X_j, gmModel)
            #sample_list, q_star_list = filterout_samples_out_of_range(integ_range, sample_list, q_star_list)# filter out samples out of the integration range
            #sample, q_star = filterout_samples_out_of_range(integ_range, X_i, q_star)
            return PMCoutput(integ_i, COV_i, integ_list, COV_list, gmModel, sample_list, q_star_list, X_i, q_star_i)

        ## replace ##
        X_i, qX_i = X_j, qX_j

    #integ_wt, COV_wt = weighted_average_integ(integ_list, stdev_list, COV_list)
    print_output(i, integ_i, COV_i, display, Ndim, integ_range, X_i, X_j, gmModel)
    #sample_list, q_star_list = filterout_samples_out_of_range(integ_range, sample_list, q_star_list)
    #sample, q_star = filterout_samples_out_of_range(integ_range, X_i, q_star)
    
    return PMCoutput(integ_i, COV_i, integ_list, COV_list, gmModel, sample_list, q_star_list, X_i, q_star_i)
    
    
    
    
def initialize(integ_range, init_gmModel, Nsmpl, Ngaussian, correlation=True):
    Ndim = len(integ_range)
    ks_stat = np.full(Ndim, 10.0)

    ## Set initial mean vector, cov matrix, and weights
    gmmean = np.stack([np.mean(integ_range, axis=1)] * Ngaussian, axis=0)  if init_gmModel == '' else init_gmModel.means_
    gmwt = np.full(Ngaussian, 1/Ngaussian) if init_gmModel == '' else init_gmModel.weights_

    #gmmean = np.array([np.random.uniform(low, high, Ngaussian) for low, high in integ_range]).T
    
    if correlation==True:
        covtype = 'full'
        gmcov = [np.diag(np.diff(integ_range)[:, 0]) for _ in range(Ngaussian)] if init_gmModel == '' else init_gmModel.covariances_        
    else:
        covtype = 'diag'
        gmcov = [np.diff(integ_range)[:, 0] for _ in range(Ngaussian)] if init_gmModel == '' else init_gmModel.covariances_

    gm_model = GaussianMixture(
        n_components=Ngaussian,
        covariance_type=covtype,
        )
    
    gm_model = set_gmm(gm_model, gmmean, gmwt, gmcov)
    
    X, label = gm_model.sample(n_samples=Nsmpl)
    qX = np.exp(gm_model.score_samples(X))

    return Ndim, ks_stat, gm_model, X, qX
    '''