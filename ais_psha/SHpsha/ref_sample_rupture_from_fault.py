import numpy as np

def M2A(M): ## A in km2
    return 10**(M - 4)
def A2M(A): ## A in km2
    return np.log10(A) + 4
def M2W(M): ## W in km
    return 10**(0.5*M - 2.15)
def W2M(W): ## W in km
    return 2. * (np.log10(W) + 2.15)
def M2L(M): ## L in km
    return 10**(0.5*M - 1.85)
def L2M(L): ## L in km
    return 2. * (np.log10(L) + 1.85)

def Sample_X_randomly_from_horizontal_vertical_fault(M,X):
    f_len = X[1] - X[0]
    rup_len = M2L(M)

    rup_len = np.where(rup_len > f_len, f_len, rup_len)

    x_left_sample = np.random.uniform(0, f_len - rup_len) + X[0]
    x_right_sample = x_left_sample + rup_len

    return np.c_[x_left_sample, x_right_sample]

def Sample_Z_randomly_from_horizontal_vertical_fault(M,Z):
    """
    Z : depth range in km (e.g., [0,12] for 0 to 12 km)
    """
    f_width = Z[1] - Z[0]
    rup_width = M2W(M)

    rup_width = np.where(rup_width > f_width, f_width, rup_width)

    z_upper_sample = np.random.uniform(0, f_width - rup_width) + Z[0]
    z_lower_sample = z_upper_sample + rup_width

    return np.c_[z_upper_sample, z_lower_sample]

def Sample_X_regularly_from_horizontal_vertical_fault(M: float, X, dX: float):######## mode ######
    f_len = np.abs(X[1] - X[0])
    rup_len = M2L(M)

    rup_len = np.where(rup_len > f_len, f_len, rup_len)
    x_left_sample = np.arange(0, f_len, dX)
    x_right_sample = x_left_sample + rup_len   
    
    mask = x_right_sample <= f_len
    x_left_sample_sel = x_left_sample[mask] + X[0]
    x_right_sample_sel = x_right_sample[mask] + X[0]

    return np.c_[x_left_sample_sel, x_right_sample_sel]

def Sample_Z_regularly_from_horizontal_vertical_fault(M: float, Z, dZ: float):
    f_width = Z[1] - Z[0]
    rup_width = M2W(M)

    rup_width = np.where(rup_width > f_width, f_width, rup_width)
    z_upper_sample = np.arange(0, f_width, dZ)
    z_lower_sample = z_upper_sample + rup_width   
    
    mask = z_lower_sample <= f_width
    z_upper_sample_sel = z_upper_sample[mask] + Z[0]
    z_lower_sample_sel = z_lower_sample[mask] + Z[0]

    return np.c_[z_upper_sample_sel, z_lower_sample_sel]

def Calc_x_direction_pdf_pmf(M, xrange, Dx):
    regular_sample_x = Sample_X_regularly_from_horizontal_vertical_fault(M, xrange, Dx)
    p = np.full_like( regular_sample_x[:,0], 1/len(regular_sample_x[:,0]) )
    pmf = p
    pdf = p / Dx
    return np.c_[regular_sample_x, pdf, pmf]

def Calc_z_direction_pdf_pmf(M, zrange, Dz):
    regular_sample_z = Sample_Z_regularly_from_horizontal_vertical_fault(M, zrange, Dz)
    p = np.full_like( regular_sample_z[:,0], 1/len(regular_sample_z[:,0]) )
    pmf = p
    pdf = p / Dz
    return np.c_[regular_sample_z, pdf, pmf]

def Gen_xyz_sample(xsmpl, zsmpl, y):
    """
    generate rupture sample using x and z samples when the fault is horizontally running and vertical.
    
    Return : rectangular rupture samples
        [ [x1, x2, y1, y2, z1, z2],
          [x1, x2, y1, y2, z1, z2],
          ...]
    """
    
    array1_repeated = np.repeat(xsmpl, zsmpl.shape[0], axis=0)
    array2_tiled = np.tile(zsmpl, (xsmpl.shape[0], 1))
    xzsmpl = np.hstack((array1_repeated, array2_tiled))

    xyzsmpl = np.insert(xzsmpl, 2, y, axis=1)
    xyzsmpl = np.insert(xyzsmpl, 3, y, axis=1)
    
    return xyzsmpl


def Dist_from_3Dpoint_to_3Dline(coord_line, coord_point): ######### MODIFY THIS CODE!!!!!!!!!!!!!!!!!!!!! ########
    """
    coord_line : [ [x1, x2, y1, y2, z1, z2],
                    [x1, x2, y1, y2, z1, z2],
                    ...
                ]
    coord_point : [x0, y0, z0]
    """
    
    #print(coord_line)
    
    x1 = coord_line[:,0]
    x2 = coord_line[:,1]
    y1 = coord_line[:,2]
    y2 = coord_line[:,3]
    z1 = coord_line[:,4]
    z2 = coord_line[:,5]
    
    x0 = coord_point[0]
    y0 = coord_point[1]
    z0 = coord_point[2]
    
    u = np.c_[ x1-x0, y1-y0, z1-z0]
    v = np.c_[ x2-x0, y2-y0, z1-z0]
    
    K = np.array( [ -np.sum(v*(u-v), axis=1) / np.sum((u-v)**2, axis=1) ] ).T
    dist1 = ( np.sum( ( u*K + v*(1-K) )**2 , axis=1 ) ) **0.5 
    dist2 = np.sum( u**2 , axis=1)**0.5
    dist3 = np.sum( v**2 , axis=1)**0.5
    
    data = np.c_[K, dist1, dist2, dist3]
    
    #print(dist1, dist2, dist3)
    
    dist = np.where((data[:, 0] >= 0) & (data[:, 0] <= 1),
                        data[:, 1],
                        np.minimum(data[:, 2], data[:, 3])) 

    return dist

def Calc_horizontal_fault_distance_pdf_pmf(target_distance, DR, M, DM, Rlist_search, Mlist_search, Rpdflist, Rpmflist):
    Rlist_search_expanded = Rlist_search[:, np.newaxis]
    condition = (Rlist_search_expanded - DR/2 <= target_distance) & (target_distance < Rlist_search_expanded + DR/2)
    R_indice = np.argmax(condition, axis=0)
    R_indice[~condition.any(axis=0)] = -1
    
    Mlist_search_expanded = Mlist_search[:, np.newaxis]
    condition = (Mlist_search_expanded - DM/2 <= M) & (M < Mlist_search_expanded + DM/2)
    M_indice = np.argmax(condition, axis=0)
    M_indice[~condition.any(axis=0)] = -1
    
    pdf_values = Rpdflist[R_indice, M_indice]
    pmf_values = Rpmflist[R_indice, M_indice]

    return pdf_values, pmf_values



    '''Mkeys = [f"{mag:.2f}" for mag in M]

    for i, Mkey in enumerate(Mkeys):
        Rpdfpmf = Rpdflist[Mkey]
        distances = Rpdfpmf[:, 0]
        diff = np.abs(distances - target_distance[i])
        index = diff <= DR/2

        if np.any(index):
            pdf, pmf = Rpdfpmf[index, 1:3][0]  # Assuming only one match is expected, otherwise this needs adjustment
        else:
            pdf, pmf = 0, 0

        pdf_values[i] = pdf
        pmf_values[i] = pmf'''

    


def Calc_horizontal_fault_distance_pdf_pmf_vectorized(target_distance, DR, M, Rpdflist):
    # Prepare arrays for pdf and pmf values
    pdf_values = np.zeros_like(M)
    pmf_values = np.zeros_like(M)

    # Vectorized computation
    for mag, Rpdfpmf in Rpdflist.items():
        # Find the indices in M that match the current magnitude
        mag_indices = np.where(np.isclose(M, float(mag), rtol=1e-2))

        # Extract the corresponding target distances
        target_distances_mag = target_distance[mag_indices]

        # Compute the distance condition for each target distance
        distances = Rpdfpmf[:, 0]
        index_matrix = np.abs(distances - target_distances_mag[:, None]) <= DR / 2

        # Find the first matching index for each target distance
        first_match_indices = np.argmax(index_matrix, axis=1)
        matches = index_matrix[np.arange(len(target_distances_mag)), first_match_indices]

        # Extract pdf and pmf values
        pdfs, pmfs = Rpdfpmf[first_match_indices, 1], Rpdfpmf[first_match_indices, 2]

        # Assign to the output arrays
        pdf_values[mag_indices] = np.where(matches, pdfs, 0)
        pmf_values[mag_indices] = np.where(matches, pmfs, 0)

    return pdf_values, pmf_values
