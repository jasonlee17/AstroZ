#!/usr/bin/env python
# coding: utf-8

#Calculate chi2 for marginalization + redshift estimation, splitting sims into small bits (1000) so calculation time is reasonable ~ 20 minutes
#Nov 14th, 2023 - Now using ugrizY sims

#Jan. 11th, 2024 => Using lcfit_TOBS_v240111.txt 3 iterations for LCFIT+z

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
import sys
import os
import pandas as pd

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord, AltAz, EarthLocation
from astropy.time import Time
from astropy.coordinates import Angle

from scipy import interpolate

import galsim
from scipy import ndimage

from scipy.interpolate import interp1d
from scipy.integrate import trapz

time_beginning = time.time()

no_split, split_ind = 800, int(sys.argv[1])

type_ind = int(sys.argv[2])
sim_types = ['REALISTIC_v1113/', 'SYST_ONLY_v1113/', 'PERFECT_v1113/']#, 'PERFECT_v1113/']
poly_types = np.array([[0.005, 0.66, 0.1], [0.005, 0.0, 0.0], [0.005, 0.0, 0.0]])
poly_vals = poly_types[type_ind]

ideal_tobs = int(sys.argv[3])
tobs_str = np.array(['realistic', 'ideal', 'lcfit'])[ideal_tobs]

wrt_ind = int(sys.argv[4])

wrt_band_str = ['Y', 'z']
    
earth_location = EarthLocation.of_site('La Silla Observatory') #ctio for DES

print(earth_location)

LSST_bands = ['u', 'g', 'r', 'i', 'z', 'Y']

ref_star_SED = np.loadtxt('/global/cfs/cdirs/des/jlee/SN_Ia/star_SEDs/ukk5v.dat')

SNANA_lsst_bands = []
for i in range(len(LSST_bands[:-1])):
    SNANA_lsst_bands.append(np.loadtxt('DCR_AstroZ/filter_functions/LSST_baseline_1.9/LSST_' + LSST_bands[i] + '.dat', skiprows = 7))

#lambda is in Angstroms for SNANA, nm for original DES bands
def effective_lambda(filt, source):
    wavelengths, filt_response = filt.transpose()[0], filt.transpose()[1]
    spacing = wavelengths[1] - wavelengths[0]
    f_source = interp1d(source.transpose()[0], source.transpose()[1])
    eff_lambda_ref = trapz(filt_response*f_source(wavelengths)*wavelengths, dx = spacing)/trapz(filt_response*f_source(wavelengths), dx = spacing)
    return eff_lambda_ref

def DCR_shift(z_a, filt, source, pressure=77.54, temperature=279.95, H2O_pressure=0.133322):
    wavelengths, filt_response = filt.transpose()[0], filt.transpose()[1]
    spacing = wavelengths[1] - wavelengths[0]
    f_source = interp1d(source.transpose()[0], source.transpose()[1])
    n = galsim.dcr.air_refractive_index_minus_one(wavelengths/10, pressure=pressure, temperature=temperature, H2O_pressure=H2O_pressure) + 1
    R0 = (n**2 - 1)/(2*n**2)
    R = R0 * np.tan(z_a)
    num, denom = trapz(R*filt_response*f_source(wavelengths), dx = spacing), trapz(filt_response*f_source(wavelengths), dx = spacing)
    return num/denom

def sky_to_alt_az_coords(SN_fid_pos, mjd, loc = earth_location):
    coord0 = SkyCoord(SN_fid_pos[0] * u.deg, SN_fid_pos[1] * u.deg)
    aa = AltAz(location=loc, obstime=Time(mjd, scale = 'utc', format = 'mjd', location = loc))
    alt, az = coord0.transform_to(aa).alt, coord0.transform_to(aa).az 
    return alt, az


def find_nearest(array, values):
    array = np.asarray(array)
    idxs = np.zeros(len(values))
    for i in range(len(values)):
        idxs[i] = (np.abs(array - values[i])).argmin()
    idxs = idxs.astype(int)
    return idxs, array[idxs]


type_str = 'JASON_DCR_ASTROZ_COADD_%s' %sim_types[type_ind]
SIM_dir_str = '/pscratch/sd/a/astjason/DCR_AstroZ/' + type_str
t_head = Table.read(glob.glob(SIM_dir_str + '*HEAD.FITS*')[0])
t_phot = Table.read(glob.glob(SIM_dir_str + '*PHOT.FITS*')[0])
t_DCR = pd.read_csv(glob.glob(SIM_dir_str + '*.DCR*')[0], delim_whitespace=True, skiprows = 6)
t_DUMP = pd.read_csv(glob.glob(SIM_dir_str + '*.DUMP*')[0], delim_whitespace=True, skiprows = 5)
#t_spec1 = Table.read(glob.glob(SIM_dir_str + '*SPEC.FITS*')[0], hdu = 1)
#t_spec2 = Table.read(glob.glob(SIM_dir_str + '*SPEC.FITS*')[0], hdu = 2)

#DCR properties: CID, MJD, BAND, LAMAVG_SED_WGTED, SNR, PSF_FWHM, TOBS, AIRMASS, SIM_DCR are important

where_u, where_g, where_r = np.where(t_DCR['BAND'] == 'u')[0], np.where(t_DCR['BAND'] == 'g')[0], np.where(t_DCR['BAND'] == 'r')[0]
where_i, where_z, where_Y = np.where(t_DCR['BAND'] == 'i')[0], np.where(t_DCR['BAND'] == 'z')[0], np.where(t_DCR['BAND'] == 'Y')[0]

#DCR properties: CID, MJD, BAND, LAMAVG_SED_WGTED, SNR, PSF_FWHM, TOBS, AIRMASS, SIM_DCR are important

epoch_range = np.arange(-18, 51)
z_range = np.arange(0.00, 1.21, 0.01) #Need to change this later to up to 1.20
AM_range = np.arange(1.00, 3.00, 0.01)
x1_vals = np.arange(-3.0, 2.5, 0.5)
c_vals = np.arange(-0.3, 0.55, 0.05)

shifts_by_x1_c = np.zeros([len(x1_vals), len(c_vals), len(SNANA_lsst_bands), len(z_range), len(epoch_range), len(AM_range)])

for x1 in range(len(x1_vals)):
    for c in range(len(c_vals)):
        shifts_by_x1_c[x1][c] = np.load('/pscratch/sd/a/astjason/DCR_AstroZ/model/LSST/SALT3.P18-NIR/interp1d/SALT3_calc_SEDs_x1_%.1f_c%.2f.npy' %(x1_vals[x1], c_vals[c]))

t_DCR_inds_CID = np.load('DCR_AstroZ/' + type_str + 't_DCR_inds_CID.npy', allow_pickle=True)
t_DCR_inds_band_CID = np.load('DCR_AstroZ/' + type_str + 't_DCR_inds_band_CID.npy', allow_pickle=True)

where_bands = [where_u, where_g, where_r, where_i, where_z, where_Y]

#Run following block once you have t_DCRs[0], alt_shifts_wrt_z_band etc
t_DCR = pd.read_csv('/pscratch/sd/a/astjason/DCR_AstroZ/' + type_str + type_str[:-1] + '_with_band_obs.csv')

def sky_to_alt_az_shift(DCR_shift_RA, DCR_shift_DEC, SN_fid_pos, mjd, loc = earth_location):
    coord0 = SkyCoord(SN_fid_pos[0] * u.deg, SN_fid_pos[1] * u.deg)
    aa = AltAz(location=loc, obstime=Time(mjd, scale = 'utc', format = 'mjd', location = loc))
    coord = SkyCoord((coord0.ra.value + DCR_shift_RA) * u.deg, (coord0.dec.value + DCR_shift_DEC) * u.deg)
    alt_shift, az_shift = coord.transform_to(aa).alt - coord0.transform_to(aa).alt, coord.transform_to(aa).az - coord0.transform_to(aa).az
    return alt_shift, az_shift


z_true = t_head['REDSHIFT_FINAL']

def sigma_stat(PSF_FWHM, SNR, poly = poly_vals[1:]):
    return poly[0]*(PSF_FWHM/SNR) + poly[1]*(PSF_FWHM/SNR)**2

sigma_syst = poly_vals[0]

#Obtaining chi2

chi2_z = np.zeros([no_split, len(LSST_bands[:-1]), len(z_range), len(x1_vals), len(c_vals)])
#SNRs_RMS_by_SNID_band = np.zeros([len(t_heads[0]), len(LSST_bands)])

z_range = np.arange(0.00, 1.21, 0.01)
no_redshifts = len(z_range)

print(time.time() - time_beginning)
time_st_total = time.time()

if ideal_tobs == 0:
    tobs_obs = np.loadtxt('/pscratch/sd/a/astjason/DCR_AstroZ/' + type_str + 'TOBS_fmax_clump.txt')
elif ideal_tobs == 1:
    tobs_obs = t_DCR['TOBS'].values
elif ideal_tobs == 2:
    tobs_obs = np.loadtxt('/pscratch/sd/a/astjason/DCR_AstroZ/' + type_str + 'lcfit_TOBS_v240111.txt')
    lcfit_inds = np.loadtxt('/pscratch/sd/a/astjason/DCR_AstroZ/' + type_str + 'lcfit_CIDs_v240111.txt')

if wrt_ind == 1:
    alt_shifts_wrt_constant_band = np.load('/pscratch/sd/a/astjason/DCR_AstroZ/' + type_str + 'alt_shift_obs_all_wrt_z_avg.npy')
if wrt_ind == 0:
    alt_shifts_wrt_constant_band = np.load('/pscratch/sd/a/astjason/DCR_AstroZ/' + type_str + 'alt_shift_obs_all_wrt_Y_avg.npy')

    
for head_ind in range(split_ind * no_split, (split_ind + 1) * no_split):
    chi2_ind = int(head_ind - split_ind * no_split)
    if ideal_tobs == 2:
        if head_ind not in lcfit_inds:
            continue
    SNID_sim = int(t_head[head_ind]['SNID'])
    t_DCR_inds_where_CID = np.where(t_DCR['CID'] == SNID_sim)[0]
    t_DCR_band_inds_used = []
    for i in range(len(LSST_bands[:-1])):
        t_DCR_band_inds_used.append(np.intersect1d(where_bands[i], t_DCR_inds_where_CID))
    t_DCR_tobs = tobs_obs
    #For one SN candidate
    
    for i in range(len(LSST_bands[:-1])):
        AM_inds = find_nearest(AM_range, t_DCR['AIRMASS'].values[t_DCR_band_inds_used[i]])[0]
        SNRs = t_DCR['SNR'].values[t_DCR_band_inds_used[i]]
        PSF_FWHMs = t_DCR['PSF_FWHM'].values[t_DCR_band_inds_used[i]]

        if len(t_DCR_band_inds_used[i]) == 0:
            #print(LSST_bands[i])
            chi2_z[chi2_ind][i] = 0.
            continue 
        shifts_obs = np.array(alt_shifts_wrt_constant_band[t_DCR_band_inds_used[i]]*3600)
        for z in range(len(z_range)):
            epochs = np.round((t_DCR_tobs[t_DCR_band_inds_used[i]])/(1+z_range[z])).astype(int)
            epochs_ind = find_nearest(epoch_range, epochs)[0]
            for x1 in range(len(x1_vals)):
                for c in range(len(c_vals)):
                    shifts_wrt_avg = shifts_by_x1_c[x1][c][i][z][epochs_ind, AM_inds] 
                    sig_total2 = sigma_stat(PSF_FWHMs, SNRs)**2 + sigma_syst**2 #sig_total = sig_stat + sig_syst
                    chi2_z[chi2_ind][i][z][x1][c] = np.sum((shifts_wrt_avg - shifts_obs)**2/sig_total2)
    if head_ind % 10 == 0:
        print(time.time() - time_st_total, head_ind)
    
np.save('/pscratch/sd/a/astjason/DCR_AstroZ/' + type_str + 'chi2_z_wrt_%s_%d_%s_tobs.npy' %(wrt_band_str[wrt_ind], split_ind, tobs_str), chi2_z)
                        
print('Done', time.time() - time_beginning)
