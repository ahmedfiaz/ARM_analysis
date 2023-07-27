
'''
PURPOSE: To run the plume model on moisture (specific humidity)
         and temperature inputs and to output plume virtual temp.
                  
AUTHOR: Fiaz Ahmed

DATE: 08/27/19
'''

import numpy as np
from netCDF4 import Dataset
from glob import glob
import datetime as dt
from dateutil.relativedelta import relativedelta
import time
import itertools
from sys import exit
from numpy import dtype
from thermodynamic_functions import *
from parameters import *
from thermo_functions import plume_lifting,calc_qsat,invert_theta_il
from scipy.interpolate import interp1d
import scipy.io as sio
import xarray as xr
from scipy.interpolate import interp1d
import os

start_time=time.time()

#### Set mixing preference ####
# Deep-inflow B mixing like in Holloway & Neelin 2009, JAS; Schiro et al. 2016, JAS
MIX='DIB'       

# No mixing case
# MIX='NOMIX'   

print('READING AND PREPARING FILES')

####### LOAD temp. & sp.hum DATA ########

dir='/home/fiaz/DOE_ASR_2022/files/'
dir_out='/home/fiaz/DOE_ASR_2022/files/'

#file_in='GoAmazon_pooled.nc'
#file_in='SGP_pooled.nc'
file_in='CACTI_pooled.nc'

# fil_out='plume_properties_ARM_oceans'
#fil_out='plume_properties_GoAmazon'
# fil_out='plume_properties_SGP'
fil_out='plume_properties_CACTI'

ds=xr.open_dataset(dir+file_in)
temp=ds['temperature']

# mixing_ratio=ds['r']*1e-3 ### Convert from g/kg -> kg/kg 
sphum=ds['sphum']
prc=ds['precip']
pres_var='p'
pres_var='pressure'

# print(sphum.max(),sphum.min())
# exit()


### Drop the Nans ###
# temp=temp.dropna(dim='time')
# sphum=sphum.dropna(dim='time')

assert(all(temp.time==sphum.time))

time_dim=temp['time'].size
lev_lowres=temp[pres_var].astype('int').values

### Get time dimension ###
time_dim=np.arange(temp.shape[0])

## Create high-res vertical levels for interpolation
#lev=np.arange(1000,445,-5)
# lev=np.arange(975,445,-5)
lev=np.arange(875,445,-5)

lev1=lev_lowres[lev_lowres<=445]    
lev=np.concatenate((lev,lev1))
lev=np.int_(lev)

dp=np.diff(lev)*-1
i450=np.where(lev==450)[0]
i600=np.where(lev==600)[0]
# i1000=np.where(lev==1000)[0]
# i1000=np.where(lev==975)[0]
i1000=np.where(lev==750)[0]

ilev=i600[0]
ibeg=i1000[0]
#ibeg=np.where(lev==700)[0][0]

## Interpolate the temperature and sp.humidity information ###

f=interp1d(lev_lowres,temp.values,axis=1,bounds_error=False)
T_interp=f(lev)

f=interp1d(lev_lowres,sphum.values,axis=1,bounds_error=False)
q_interp=f(lev)


## Launch plume from 995 hPa ##
ind_launch=np.zeros((time_dim.size),dtype='int')
ind_launch[:]=i1000+1
        
### Prescribing mixing coefficients ###
c_mix_DIB = np.zeros((time_dim.size,lev.size)) 

## Compute Deep Inflow Mass-Flux ##
## The mass-flux (= vertical velocity) is a sine wave from near surface
## to 450 mb. Designed to be 1 at the level of launch
## and 0 above max. w i.e., no mixing in the upper trop.
assert(all(ind_launch>=1))
w_mean = np.sin(np.pi*0.5*(lev[ind_launch-1][:,None]-lev[None,:])/(lev[ind_launch-1][:,None]-lev[i450]))    
minomind = np.where(w_mean==np.nanmax(w_mean))[0][0]
c_mix_DIB = np.zeros((time_dim.size,lev.size)) ## 0 above max. w
c_mix_DIB[:,1:-1]= (w_mean[:,2:] - w_mean[:,:-2])/(w_mean[:,2:]+w_mean[:,:-2])
c_mix_DIB[c_mix_DIB<0]=0.0
c_mix_DIB[np.isinf(c_mix_DIB)]=0.0
c_mix_NE = np.zeros((time_dim.size,lev.size)) ## Non-entraining mixing coefficients


### Change data type ####
temp_fin=np.float_(T_interp)
sphum_fin=np.float_(q_interp)
    
### Set output variables ####
temp_plume=np.zeros_like(temp_fin)    
temp_v_plume=np.zeros_like(temp_fin)    

temp_plume_NE=np.zeros_like(temp_fin)    
temp_v_plume_NE=np.zeros_like(temp_fin)    

# for i in np.arange(lev.size):
#     print(lev[i],c_mix_DIB[0,i],temp_fin[0,i],sphum_fin[0,i])

## Launch plume ###
print('DOING DIB PLUME COMPUTATION')
plume_lifting(temp_fin, sphum_fin, temp_v_plume, temp_plume, 
c_mix_DIB, lev, ind_launch)

print('DOING Non-Entraining PLUME COMPUTATION')
plume_lifting(temp_fin, sphum_fin, temp_v_plume_NE, temp_plume_NE, 
c_mix_NE, lev, ind_launch)


#### Optional thermodynamic computations ###

# qsat ##
# qsat=qs_calc(lev, temp)
# rh=(sphum_fin/qsat)*100.
# 
## env. virtual temp. ###
temp_v_env = temp_v_calc(T_interp, q_interp, 0.) ### Environmental virtual temp.

print(temp_v_plume.shape)

## thermal buoyancy ####
def compute_buoy_temp(temp_v_plume,temp_v_env,ibeg,ilev,lev):
    '''
    ibeg and ilev are the level indices for vertical averaging.
    '''
    
    tvdiff=g*(temp_v_plume-temp_v_env)/temp_v_env
    tvdiff[temp_v_plume==0]=np.nan

    var=np.copy(tvdiff[:,ibeg:ilev])
    var1=np.copy(temp_v_env[:,ibeg:ilev])
    var[var1==0]=np.nan

    ind=np.where(np.isfinite(var))[1]
    lev_trunc=lev[ibeg:ilev+1]
    istrt=ind[0] ## start from first finite value
    buoy_integ=np.nansum(var[:,istrt:ind[-1]]*dp[None,istrt:ind[-1]],axis=1)/(lev_trunc[istrt]-lev_trunc[ind[-1]])
    return buoy_integ
    
    
buoy_integ=compute_buoy_temp(temp_v_plume,temp_v_env,ibeg,ilev,lev)
buoy_integ_NE=compute_buoy_temp(temp_v_plume_NE,temp_v_env,ibeg,ilev,lev)

################################################

print ('SAVING FILE')
fout=dir_out+fil_out+'.nc'
data_set=xr.Dataset(data_vars={'Tv_plume_DIB': (('time','lev'), temp_v_plume),
                               'Tv_plume_NE': (('time','lev'), temp_v_plume_NE),
                               'Tv_env': (('time','lev'), temp_v_env),
                               'BL': (('time'), buoy_integ),
                               'BL_NE': (('time'), buoy_integ_NE)},
                    coords={'time':temp.time,
                            'lev':lev,
                            })

data_set.Tv_plume_DIB.attrs['units']="K"
data_set.Tv_plume_NE.attrs['units']="K"

data_set.Tv_plume_DIB.attrs['description']="Plume virtual temperatures from a deep-inflow mixing scheme"
data_set.Tv_plume_NE.attrs['description']="Plume virtual temperatures without entrainment"

data_set.Tv_env.attrs['units']="K"

data_set.BL.attrs['units']="m/s^2"
data_set.BL_NE.attrs['units']="m/s^2"

data_set.attrs['source']="Plume model output from ARM TWP sites"

### Manually clobbering since .to_netcdf throws permission errors ###

try:
    os.remove(fout)
except:
    pass

data_set.to_netcdf(fout,
mode='w',
engine='netcdf4')


print ('FILE WRITTEN')
print ('TOOK %.2f MINUTES'%((time.time()-start_time)/60))
