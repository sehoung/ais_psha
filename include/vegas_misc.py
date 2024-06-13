import numpy as np
import matplotlib.pyplot as plt
def Run_vegas_figure(Bdr_hist, integ_range, integ_hist, CV_hist, sus_flag, Ngrid):
    Niter=len(integ_hist)-1
    aep_hist = 1 - np.exp(-integ_hist)
    print("AEP:",aep_hist[-2])
    print("COV:",CV_hist[-2])
    
    fig, ax = plt.subplots(4, figsize=(4,8))
    
    fig.suptitle("Deaggregation")
    for i_dim in range(0, len(integ_range)):
        plot_arr = np.arange(0,Ngrid[i_dim]*2,1)
        
        i=Niter-1
        boundaries = Bdr_hist[i][i_dim]
        values = 1/np.diff(Bdr_hist[i][i_dim])/Ngrid[i_dim]
        x_values = []
        y_values = []
        for j in range(len(values)):
            x_values.extend([boundaries[j], boundaries[j+1]])
            y_values.extend([values[j], values[j]])
        plot_arr = np.c_[plot_arr, np.c_[x_values, y_values]]
        
        ax[i_dim].plot(plot_arr[:,1], plot_arr[:,2])
    fig.tight_layout()
    plt.show()
    