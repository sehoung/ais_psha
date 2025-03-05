import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import entropy
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import sys 
from collections import namedtuple

PMCoutput = namedtuple('PMCoutput', ['integ', 'mean', 'cov', 'sample', 'q_star'])

# version 0_1
# add output sample and optimal density



def Run_pmc_normal_integ(f, integ_range, Nsmpl, Niter=10, f_arg=[], KS_statistic_threshold=0.1, display=False):
    """
    f : integrand
    integ_range : integration range
    Nsmpl : number of samples
    """

    # Initialize #
    Ndim = len(integ_range)
    ks_stat = np.full(Ndim, 10.0)

    mean = np.mean(integ_range, axis=1)
    cov = (np.diag(np.diff(integ_range)[:,0]))
    mvn = multivariate_normal(mean = mean, cov = cov)
    X = mvn.rvs(size=Nsmpl)
    qX = mvn.pdf(X)
    
    for i in range(Niter):
        # Calc functional value
        fX = f(X, arg=f_arg)
        if i==0:
            sample_all = X
            fX_all = fX
        else:
            sample_all = np.vstack((sample_all, X))
            fX_all = np.hstack((fX_all, fX))

        ## update sample pdf ##
        wt = fX/qX
        new_pdf =  wt/np.sum(wt)

        ## update parameters ##
        X_new = X[ np.random.choice(len(X), size=len(X), p=new_pdf) ]
        mean_new = np.mean(X_new, axis=0)
        cov_new = np.cov(X_new, rowvar=False)

        ## update pdf ##
        mvn_new = multivariate_normal(mean = mean_new, cov = cov_new)

        ## resample from updated pdf ##
        X_new = mvn_new.rvs(size=Nsmpl)

        ## update sample pdf ##
        qX_new = mvn_new.pdf(X_new)
        
        ## stopping criteria: if sample pdf does not change, STOP ##
        for i in range(Ndim):
            ks_stat[i] = ks_2samp(X[:,i], X_new[:,i])[0]

        #if pdf_diff_mean_change < 0.01 and pdf_diff_stdev_change < 0.01 :
        if np.all(ks_stat < 0.05):
            integ = np.mean(wt)
            stdev = np.sqrt( (np.mean(wt*wt) - integ**2) /Nsmpl )
            COV = stdev/integ*100
            
            print("converged at i = ", i, ","
              , "Integ = ", "{:.12e}".format(integ)
              ,  "COV = ", "{:.2f}".format(COV), "%"
              , file=sys.stderr)
            ## show the result figure ##
            if display == True:
                fig, ax = plt.subplots(Ndim)
                for i in range(Ndim):
                    ax[i].set_xlim(integ_range[i,:])
                    ax[i].hist(X[:,i], label="old", density=True, alpha=1.0, bins=20, color='gray')
                    ax[i].hist(X_new[:,i], label="new", density=True, alpha=0.5, bins=20, color='orange')
                    ax[i].plot(np.sort(X_new[:,i]), norm.pdf(np.sort(X_new[:,i]), loc=mean_new[i], scale=cov_new[i,i]**0.5), color='orange')
                    ax[i].legend()
                fig.tight_layout()
                plt.show()
            ###

            q_star_all = fX_all / integ
            return PMCoutput(integ, mean, cov, sample_all, q_star_all)

        ## replace ##
        X, qX, mean, cov = X_new, qX_new, mean_new, cov_new

    q_star_all = fX_all / integ
    return PMCoutput(integ, mean, cov, sample_all, q_star_all)


def Run_pmc_figure(pmc_output, integ_range,num_bins=100, method='nearest'):
    Ndim = len(integ_range)
    mean = pmc_output.mean
    cov = pmc_output.cov
    sample = pmc_output.sample
    q_star = pmc_output.q_star
    if Ndim ==2:
        fig_optimal_all, ax_optimal_all = plt.subplots()
        scatter=ax_optimal_all.scatter(sample[:,0],sample[:,1], c=q_star, s=1, cmap='rainbow')
        colorbar=plt.colorbar(scatter)
        ax_optimal_all.set_title("Optimal density")
        ax_optimal_all.set_xlim(integ_range[0,0], integ_range[0,1])
        ax_optimal_all.set_ylim(integ_range[1,0], integ_range[1,1])
        #ax_optimal_all.set_xlim(5,6.5)
        #ax_optimal_all.set_ylim(2,4)

    marginal_pdfs_optimal, bin_centers_optimal = marginalize_pdf(sample, q_star, integ_range, num_bins, method)
    #### plot optimal density
    fig, ax = plt.subplots(Ndim)
    for i_dim in range(0, Ndim): # i_dimth dimention
        x_plot = bin_centers_optimal[:,i_dim]
        if i_dim == 0:
            ax[i_dim].plot(x_plot, marginal_pdfs_optimal[:,i_dim], label="optimal", color = 'red')
            ax[i_dim].plot(x_plot, norm.pdf(x_plot,loc=mean[i_dim], scale=cov[i_dim,i_dim]**0.5), label="fitted", color = 'black')
        else:
            ax[i_dim].plot(x_plot, marginal_pdfs_optimal[:,i_dim], color = 'red')
            ax[i_dim].plot(x_plot, norm.pdf(x_plot,loc=mean[i_dim], scale=cov[i_dim,i_dim]**0.5), color = 'black')
    fig.legend()
    fig.tight_layout()
    plt.show()
        #fig_optimal.colorbar(ax_optimal.contourf(grid_x, grid_y, z, cmap='rainbow'))


def marginalize_pdf(sample, pdf_values, integ_range, num_bins, method):
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
    