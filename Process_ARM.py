import xarray as xr
import matplotlib.pyplot as plt
import pathlib
from thermodynamic_functions import qs_calc, theta_e_calc
import numpy as np
import multiprocessing
from itertools import repeat

    
def read_compute_thermo(file, var_list, pbl_top, pbl_bottom):
    ds = read_process_files(file, var_list)
    if ds is None:
        return
    else:
        ds = compute_thermo(ds, pbl_top, pbl_bottom)
        return ds

def read_process_files(file, var_list):
    """
    Open file, remove NaNs 
    """
    ds=xr.open_dataset(file)[var_list]

    # remove NaNs
    tdp = ds.Td_p.dropna('time')
    T_p = ds.T_p.dropna('time')

    if tdp.time.size == 0 or T_p.time.size == 0:
        ds = None
        return 

    time_valid = np.intersect1d(tdp.time, T_p.time)
    # cond = ds.time.isin(tdp.time) and ds.time.isin(T_p.time)
    ds = ds.where(ds.time.isin(time_valid), drop = True)
    return ds
        

def compute_thermo(ds, pbl_top, pbl_bottom, level_name = 'p'):

    """
    Compute specific humidity and sat. specific humidity
    and then layer averaged thetae variables.
    """

    # convert surf. pressure to Pa.
    ds = ds.assign(p_sfc = ds["p_sfc"] * 100.)

    # compute sp.humidity and sat. sp. humidity values
    ds = ds.assign(hus = qs_calc(ds.p, ds.Td_p), 
                                qsat = qs_calc(ds.p, ds.T_p))

    # compute theta_e and theta_e_sat
    ds = ds.assign(thetae = theta_e_calc(ds.p, ds.T_p, ds.hus),
                                thetae_sat = theta_e_calc(ds.p, ds.T_p, ds.qsat))

    thetae_bl = _bl_ave(ds.thetae, pbl_top, pbl_bottom, level_name)
    thetae_lft = _lft_ave(ds.thetae, pbl_top, level_name)
    thetae_sat_lft = _lft_ave(ds.thetae_sat, pbl_top, level_name)

    ds = ds.assign(thetae_bl = thetae_bl, thetae_lft = thetae_lft, thetae_sat_lft = thetae_sat_lft)

    # drop unnecessary variables
    # ds = ds.drop(['Td_p', 'p_sfc', 'qsat'])

    return ds        

def _bl_ave(var, pbl_top, pbl_bottom, level_name):

    lev = var.p
    pbl_thickness = pbl_bottom - pbl_top

    ### get trop. contribution ###
    cond = np.logical_and(lev >= pbl_top, lev <= pbl_bottom)
    var_bl = var.where(cond)
    var_bl_contribution = _perform_layer_ave(var_bl, lev, level_name)

    return var_bl_contribution / pbl_thickness
    
def _lft_ave(var, pbl_top, level_name):

    pbl_top = pbl_top
    lev = var.p

    var_lft = var.where(lev <= pbl_top)
    lft_top_lev = xr.where(np.isfinite(var_lft), lev, np.nan).idxmin(level_name)
    lft_thickness = pbl_top - lft_top_lev
    var_lft_contribution = _perform_layer_ave(var_lft, lev, level_name)

    return var_lft_contribution / lft_thickness


def _perform_layer_ave(var, lev, lev_name):

    dp = abs(lev.diff(lev_name)).assign_coords({lev_name : np.arange(0,lev.size-1)})
    var1 = var.isel(p = slice(0,lev.size-1)).assign_coords({lev_name: np.arange(0,lev.size-1)})
    var2 = var.isel(p = slice(1,lev.size)).assign_coords({lev_name: np.arange(0,lev.size-1)})
    return ((var1+var2) * dp * 0.5).sum(lev_name)

class ProcessARM:
    def __init__(self, path, nprocs = 2, level_name = 'p', 
                 pbl_top = 900., pbl_bottom = 1000.) -> None:
        
        self.path = path
        self.nprocs = nprocs  # no of processors to be used for parallel programming
        self.files = [str(f) for f in path.glob('*')]  # get files from path
        self.files.sort()
        
        self.ds = None
        self.var_list = ['T_p','Td_p','prec_sfc','p_sfc','u_p','v_p']  # Temp, dewpoint, precip., surf. pressure, u and v winds
        self.pbl_top = pbl_top  # in hPa
        self.pbl_bottom = pbl_bottom
        self.level_name = level_name

        self.cape = None
        self.subsat = None
        self.buoy = None
    
    def __str__(self) -> str:
        return self.path.parts[-1]

    @staticmethod
    def compute_cape(ds):
        return (ds.thetae_bl - ds.thetae_sat_lft) * 340.0/ds.thetae_sat_lft

    @staticmethod
    def compute_subsat(ds):
        return (ds.thetae_sat_lft - ds.thetae_lft) * 340.0/ds.thetae_sat_lft
   
    @staticmethod    
    def compute_buoy(cape, subsat):
        return (cape - subsat) * 9.8/340.0


    def main(self):
        pool = multiprocessing.Pool(self.nprocs)
        result = pool.starmap(read_compute_thermo, 
                              zip(self.files, 
                                  repeat(self.var_list), 
                                  repeat(self.pbl_top),
                                  repeat(self.pbl_bottom)))
        result = [i for i in result if i is not None]
        ds = xr.concat(result, dim = 'time')
        self.cape = self.compute_cape(ds)
        self.subsat = self.compute_subsat(ds)
        self.buoy = self.compute_buoy(self.cape, self.subsat)
        self.precip = ds.prec_sfc







