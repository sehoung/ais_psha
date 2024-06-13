import numpy as np


def calc_pdf_dist_fault(coord_site, coord_fault, Y):
    """
    coord_site : numpy array (x, y)
    coord_fault : numpy array size  = N x 2 ( [ [x0, y0], [x1,y1], ... ] )
    Y : numpy array: target distances to get the value of pdf
    """
    Nseg = len(coord_fault)-1
    fY = np.zeros_like(Y)
    Yminmax= np.zeros(( Nseg, 2 ))
    # Calculate Pseg
    Lseg = np.sqrt(np.sum((coord_fault[1:] - coord_fault[:-1])**2, axis=1))
    Pseg = Lseg / np.sum(Lseg)
    
    for i in range(Nseg):
        f, Yminmax[i,:] = calc_pdf_dist_single_segment(coord_site, coord_fault[i,:], coord_fault[i+1,:], Y, Pseg[i] )
        fY +=f

    return fY, [ np.min(Yminmax[:,0]), np.max(Yminmax[:,1])   ]



def calc_pdf_dist_single_segment(coord_site, coord_seg1, coord_seg2, Y, Pseg):
    """
    coord_site : numpy array (x, y)
    coord_seg1 : numpy array (x, y)
    coord_seg2 : numpy array (x, y)
    Y : numpy array: target distances to get the value of pdf
    """
    x0 = coord_site[0]
    y0 = coord_site[1]
    x1 = coord_seg1[0]
    y1 = coord_seg1[1]
    x2 = coord_seg2[0]
    y2 = coord_seg2[1]

    u = np.array([x1-x0, y1-y0])
    v = np.array([x2-x0, y2-y0])

    #Y = [a*k + b , c*k + d]
    a = u[0]-v[0]
    b = v[0]
    c = u[1]-v[1]
    d = v[1]

    k_opt = -(a*b + c*d) / (a*a + c*c)

    A = a*a + c*c
    B = 2*(a*b + c*d)
    C = b*b +d*d

    Y_k_zero = C**0.5
    Y_k_one = (A+B+C)**0.5
    Y_k_opt = (A*k_opt*k_opt + B*k_opt + C)**0.5

    if k_opt>0 and k_opt<1: # when k_opt is in between 0<k<1
        Ymin = np.min(np.array([ Y_k_zero,  Y_k_one, Y_k_opt ]))
        Ymax = np.max(np.array([ Y_k_zero,  Y_k_one, Y_k_opt ]))

        if Y_k_one < Y_k_zero:
            Y_ = Y_k_one
        elif Y_k_zero < Y_k_one:
            Y_ = Y_k_zero
        elif Y_k_one == Y_k_zero:
            Y_ = Y_k_zero # or Y_k_one
        
    else: # when k_opt is out of 0<k<1        
        Ymin = np.min(np.array([ Y_k_zero,  Y_k_one ]))
        Ymax = np.max(np.array([ Y_k_zero,  Y_k_one ]))
        
        Y_ = Ymin

    determ = np.full(Y.shape, np.inf)
    mask = (Y > Ymin) & (Y <= Ymax)
    determ[mask] = B*B - 4 * A * (C - Y[mask]**2)
    dkdY = np.where(Y < Y_, 2 * np.abs(2 * Y / determ**0.5), np.abs(2 * Y / determ**0.5))

    fk = 1

    fY_given_seg = dkdY  * fk
    fY = fY_given_seg * Pseg
    return fY, [Ymin, Ymax]

'''def sample_random_points_along_fault(coordinates, num_points=100):
    """
    Samples random points along a kinked line defined by the given coordinates.

    Parameters:
    coordinates (array-like): List of [x, y] coordinates defining the kinked points.
    num_points (int): Number of random points to sample along the kinked line. Default is 100.
    plot_result (bool): If True, plot the kinked line and sampled points. Default is True.

    Returns:
    np.ndarray: Array of sampled random points of shape (num_points, 2).
    """

    # Convert list of coordinates to numpy array for easier manipulation
    coordinates = np.array(coordinates)
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]

    # Calculate the length of each segment
    segment_lengths = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)

    # Calculate the cumulative segment lengths
    cumulative_lengths = np.cumsum(segment_lengths)

    # Total length of the kinked line
    total_length = cumulative_lengths[-1]

    # Generate random points along the total length
    random_lengths = np.sort(np.random.rand(num_points) * total_length)

    # Find the segment each random length falls into
    segment_indices = np.searchsorted(cumulative_lengths, random_lengths)

    # Calculate the points along the segments
    random_points = []
    for i, seg_idx in enumerate(segment_indices):
        if seg_idx == 0:
            segment_start_length = 0
        else:
            segment_start_length = cumulative_lengths[seg_idx - 1]

        segment_fraction = (random_lengths[i] - segment_start_length) / segment_lengths[seg_idx]

        x_point = x_coords[seg_idx] + segment_fraction * (x_coords[seg_idx + 1] - x_coords[seg_idx])
        y_point = y_coords[seg_idx] + segment_fraction * (y_coords[seg_idx + 1] - y_coords[seg_idx])
        
        random_points.append([x_point, y_point])

    random_points = np.array(random_points)


    return random_points


##### Example ######
#set site location
x0 = 1
y0 = 1
site = np.array([x0, y0])

# set fault trace
Nseg = 10
fault_points = np.random.uniform(-10, 10, (Nseg+1, 2))
fault = fault_points

#Distance samples
R = np.sort(np.random.uniform(0, 20,size=10000))

# compute pdf of R
fR, [Rmin, Rmax] = calc_pdf_dist_fault(site, fault, R)


#Verification using MC
smpl = sample_random_points_along_fault(fault, num_points=100000)
d_smpl = ( (smpl[:,0]-site[0])*(smpl[:,0]-site[0]) + (smpl[:,1]-site[1])*(smpl[:,1]-site[1]) ) ** 0.5

import matplotlib.pyplot as plt
fig, ax = plt.subplots( 2,1, figsize = (8,8))
ax[0].plot(fault[:,0], fault[:,1], color = "black")
ax[0].scatter(fault[:,0], fault[:,1], color = "black")
ax[0].scatter(smpl[:,0], smpl[:,1], color = "red", s=2)
ax[0].scatter(site[0], site[1], marker='^', color = "green")
ax[0].grid(alpha=0.5)
ax[1].hist(d_smpl, bins = 100, density=True)
ax[1].plot(R, fR, linestyle="--", color="red")
ax[1].set_xlim(Rmin, Rmax)
fig.savefig("pdf.png", dpi=300)
plt.show()'''