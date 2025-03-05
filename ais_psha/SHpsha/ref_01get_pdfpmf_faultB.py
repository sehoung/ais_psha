import numpy as np
from ref_sample_rupture_from_fault import *
import matplotlib.pyplot as plt

##### PRE SETTING #########
Mmin_integ = 5.0
Mmax_integ = 7.0
Rmin_integ = 50
Rmax_integ = 70
emin_integ = -6
emax_integ = +6
dM = 0.01
dR = 0.1
de = 0.01
Mbdrlist = np.arange(Mmin_integ, Mmax_integ+dM/2, dM)
Rbdrlist = np.arange(Rmin_integ, Rmax_integ+dR/2, dR)
ebdrlist = np.arange(emin_integ, emax_integ+de/2, de)
##################################

###### FUNCTION FOR YC1985 CHARACTERISTIC EARTHQUAKE PDF AND PMF ###########
def calc_yc1985_pdf_pmf(M, sr, A, bval, mmin, mmax, dm):
    dMchar = 0.5
    Nm0, Nmc = calc_yc1985_Nm0_Nmc(sr, A, bval, mmin, mmax)
    range_exp = (M >= mmin) & (M <= mmax-dMchar)
    range_char = (M > mmax-dMchar) & (M <= mmax)
    range_invalid = (M <  mmin) | (M > mmax)

    # Initialize Mpdf with the same shape as M
    #Mpdf = np.empty_like(M)
    #Mpdf[range_exp] = formula1[range_exp]
    #Mpdf[range_char] = formula2[rhange_char]

    nM = np.empty_like(M)
    nM[range_exp] = (Nm0-Nmc) * bval * np.log(10) * 10**(-bval * (M[range_exp] - mmin)) / (1 - 10**(-bval * (mmax - mmin)))
    nM[range_char] = np.zeros_like(M[range_char]) + Nmc/dMchar
    nM[range_invalid] = 0
   
    yc1985_pdf = nM / Nm0
    yc1985_pdf #####
    yc1985_pmf = yc1985_pdf * dm
    Normfactor = np.sum(yc1985_pmf)

    yc1985_pdf = yc1985_pdf / Normfactor
    yc1985_pmf = yc1985_pmf / Normfactor


    return yc1985_pdf, yc1985_pmf, Nm0

def calc_yc1985_Nm0_Nmc(sr, A, bval, mmin, mmax):
    c=1.5
    M0rate = SR_to_M0RATE(sr, A)
    beta = bval * np.log(10)
    
    M0u = M_to_M0(mmax)    
    AA = np.exp( -beta*( mmax - mmin - 0.5 ) )
    BB = bval * 10**(-c/2.) / (c-bval)
    CC = bval * np.exp(beta) * ( 1 - 10**(-c/2.) ) / c

    X = M0rate * (1-AA) / ( AA * M0u * ( BB + CC )  ) ## X = Nm0 - Nmc (Total number of ea in exp range)
    Nmc = beta * X * AA * np.exp(beta) / ( 2 * ( 1 - AA ) ) ## Total N of eq in characteristic range
    Nm0 = X + Nmc ## Total N of eq of the whole range

    return Nm0, Nmc

def M_to_M0(M): # M0 = Nm
    M0 = 10**(9.05 + 1.5*M)
    return M0

def M0_to_M(M0): # M0 = Nm
    M = 1/1.5 * (np.log10(M0) - 9.05)
    return M

def SR_to_M0RATE(slip_rate, area):
    """
    Args:
        slip_rate (numpy array): mm/yr
        area (numpy array): km2

    Returns:
        numpy array: M0rate in Nm/yr
    """
    shear_modulus = 3*10**10 ## N/m2, == 3*10**11 dyne/cm2
    m0rate = shear_modulus * area * slip_rate * 1000
    return m0rate 
##################



def calc_area1_R_given_M_pdf_pmf(Mlist, Rbdrlist, Rlist, dR):
    Nsmpl = int(1e7)
    x = np.random.uniform(-100, 100, size=Nsmpl)
    y = np.random.uniform(-100, 100, size=Nsmpl)
    z = np.random.uniform(5, 10, size=Nsmpl)
    xyz = np.c_[x, y, z]

    indices_to_remove = np.where( xyz[:, 0]**2 + xyz[:, 1]**2 > 10000)[0]
    xyz = np.delete(xyz, indices_to_remove, axis=0)
    dist = (xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)**0.5
    Nsmpl_actual = len(dist)

    Rcount = np.zeros_like(Rlist)
    indices = np.searchsorted(Rbdrlist, dist, side='right') - 1
    np.add.at(Rcount, indices, 1)
    Rpmf = Rcount/Nsmpl_actual
    Rpdf = Rpmf / dR

    RcMpdf = np.zeros( (len(Mlist), len(Rlist)) )
    RcMpmf = np.zeros( (len(Mlist), len(Rlist)) )
    for i, M in enumerate(Mlist):
        RcMpdf[i,:] = Rpdf
        RcMpmf[i,:] = Rpmf

    return RcMpdf, RcMpmf


sid = "faultB_B"
bval = 0.9
mmin = 5.0
mmax = 7.0
sr = 2
xmin = -42.5
xmax = 42.5
zmin = 0
zmax = 12
y = 50
Dx = 0.05
Dz = 0.05
fault_area = (xmax - xmin) * (zmax - zmin)

Mlist = (Mbdrlist[:-1] + Mbdrlist[1:]) / 2
Rlist = (Rbdrlist[:-1] + Rbdrlist[1:]) / 2
elist = (ebdrlist[:-1] + ebdrlist[1:]) / 2

Mpdf, Mpmf, rate_faultC_A = calc_yc1985_pdf_pmf(Mlist, sr, fault_area, bval, mmin, mmax, dM)
print(np.sum(Mpmf))
RcMpdf = np.zeros((len(Mlist), len(Rlist)))
RcMpmf = np.zeros((len(Mlist), len(Rlist)))
xrange = np.array([xmin, xmax])
zrange = np.array([zmin, zmax])
for i, M in enumerate(Mlist):
    Mkey = f"{M:.2f}"
    Xlocprob = Calc_x_direction_pdf_pmf(M, xrange, Dx)
    Zlocprob = Calc_z_direction_pdf_pmf(M, zrange, Dz)
    xyzsmpl = Gen_xyz_sample(Xlocprob[:,0:2], Zlocprob[:,0:2],y)
    Distlist = np.sort(Dist_from_3Dpoint_to_3Dline(xyzsmpl, np.array([0,0,0])))
    lower_bounds = Rlist - dR/2
    upper_bounds = Rlist + dR/2
    counts = np.sum((lower_bounds[:, np.newaxis] <= Distlist) & (Distlist <= upper_bounds[:, np.newaxis]), axis=1)
    RcMpmf[i,:] = counts/np.sum(counts)
    RcMpdf[i,:] = RcMpmf[i,:]/dR

MRpdf = np.zeros((len(Mlist), len(Rlist)))
MRpmf = np.zeros((len(Mlist), len(Rlist)))
for i, M in enumerate(Mlist):
    MRpdf[i,:] = RcMpdf[i,:] * Mpdf[i]
    MRpmf[i,:] = RcMpmf[i,:] * Mpmf[i]

fig, ax = plt.subplots()
ax.set_aspect(0.0005)
ax.imshow(np.log(MRpdf), cmap='hot', interpolation='nearest')

ax.set_title('Probability Heatmap')
ax.set_xlabel('Index')
ax.set_ylabel('Index')
plt.show()

print(rate_faultC_A)