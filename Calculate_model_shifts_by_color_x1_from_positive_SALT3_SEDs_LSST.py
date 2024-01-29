#For LSST

import numpy as np
import math
import matplotlib.pyplot as plt
import astropy
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.table import Table
import matplotlib
import glob

import time

t_begin = time.time()

import sys
import os
import pandas as pd
import galsim

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord, AltAz, EarthLocation
from astropy.time import Time
from astropy.coordinates import Angle

from scipy import interpolate

x1_ind, c_ind = int(sys.argv[1]), int(sys.argv[2])
x1_val = np.arange(-3.0, 3.5, 0.5)[x1_ind]
c_val = np.arange(-0.30, 0.55, 0.05)[c_ind]

print("x1, c: %.2f, %.2f" %(x1_val, c_val))

epoch_range = np.arange(-18, 51)
#z_range = np.arange(0.01, 1.20, 0.01)
z_range = np.array([1.19, 1.20])
z_inds = [119, 120]

# location of CTIO
earth_location = EarthLocation.of_site('ctio')

print(earth_location)

LSST_bands = ['u', 'g', 'r', 'i', 'z']

LSST_colors = ['blue', 'green', 'red', 'orange', 'violet']

ref_star_SED = np.loadtxt('/global/cfs/cdirs/des/jlee/SN_Ia/star_SEDs/ukk5v.dat')

SNANA_lsst_bands = []
for i in range(len(LSST_bands)):
    SNANA_lsst_bands.append(np.loadtxt('DCR_AstroZ/filter_functions/LSST_baseline_1.9/LSST_' + LSST_bands[i] + '.dat', skiprows = 7))

from scipy.interpolate import interp1d
from scipy.integrate import trapz

# R \approx R_0 tan(z_a)
# R_0 = \frac{n^2 - 1}{2n^2}
# R_shift = \frac{int dlambda R(lambda, z_a) F(lambda) S(lambda)}{int dlambda F(lambda) S(lambda)}

def DCR_shift(z_a, filt, source, pressure=77.54, temperature=279.95, H2O_pressure=0.133322):
    wavelengths, filt_response = filt.transpose()[0], filt.transpose()[1]
    spacing = wavelengths[1] - wavelengths[0]
    f_source = interp1d(source.transpose()[0], source.transpose()[1], fill_value = 'extrapolate')
    n = galsim.dcr.air_refractive_index_minus_one(wavelengths/10, pressure=pressure, temperature=temperature, H2O_pressure=H2O_pressure) + 1
    R0 = (n**2 - 1)/(2*n**2)
    R = R0 * np.tan(z_a)
    num, denom = trapz(R*filt_response*f_source(wavelengths), dx = spacing), trapz(filt_response*f_source(wavelengths), dx = spacing)
    return num/denom

def find_nearest(array, values):
    array = np.asarray(array)
    idxs = np.zeros(len(values))
    for i in range(len(values)):
        idxs[i] = (np.abs(array - values[i])).argmin()
    idxs = idxs.astype(int)
    return idxs, array[idxs]

sn_SEDs_all_z = np.load('/pscratch/sd/a/astjason/DCR_AstroZ/model/SALT3.P22-NIR/SEDs/SEDs_x1_%.1f_c%.2f_9th_CL.npy' %(x1_val, c_val))
sn_SEDs_all_z = sn_SEDs_all_z.clip(min=0) #Setting all SEDs non-negative
        
AM_range = np.arange(1.00, 3.00, 0.01)
zenith_angles = np.arccos(1/AM_range)

#Try pre-calculting ref_shift 

ref_star_shifts = np.zeros([len(SNANA_lsst_bands), len(zenith_angles)])

for b in range(len(LSST_bands)):
    for i in range(len(zenith_angles)):
        ref_star_shifts[b][i] = DCR_shift(zenith_angles[i], SNANA_lsst_bands[b], ref_star_SED)
        
shifts_by_AM_analytic_SNANA_SEDs = np.zeros([len(LSST_bands), len(z_range), len(epoch_range), len(zenith_angles)])

print("Starting DCR calculation", time.time() - t_begin)

for b in range(len(LSST_bands)):
    for z in range(len(z_range)):
        if z % 50 == 0:
            print(time.time() - t_begin)
        for e in range(len(epoch_range)):
            for i in range(len(zenith_angles)):
                SN_shift = DCR_shift(zenith_angles[i], SNANA_lsst_bands[b], sn_SEDs_all_z[z_inds[z]][e])
                shifts_by_AM_analytic_SNANA_SEDs[b][z][e][i] = (SN_shift - ref_star_shifts[b][i])*206264.806
                
np.save('/pscratch/sd/a/astjason/DCR_AstroZ/model/LSST/SALT3.P18-NIR/missing_z_only/SALT2_calc_SEDs_x1_%.1f_c%.2f.npy' %(x1_val, c_val), shifts_by_AM_analytic_SNANA_SEDs)

print(time.time() - t_begin)