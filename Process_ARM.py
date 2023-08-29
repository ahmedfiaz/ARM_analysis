import xarray as xr
import matplotlib.pyplot as plt
import pathlib
from thermodynamic_functions import qs_calc, theta_e_calc
import numpy as np
import multiprocessing
from itertools import repeat

    
def read_compute_thermo(file, var_dict, pbl_thickness_ratio, 
                        pmid):
    ds = read_process_files(file, var_dict)
    if ds is None:
        return
    else:
        ds = compute_thermo(ds, pbl_thickness_ratio, pmid)
        return ds

def read_process_files(file, var_dict):
    """
    Open file, remove NaNs 
    """
    var_list = list(var_dict.values())
    ds = xr.open_dataset(file)[var_list]

    var_dict_inv = {k:v for v,k in var_dict.items()}  # invert dictionary with key:value pairs
    ds = ds.rename(**var_dict_inv) # rename variables

    # remove NaNs
    tdp = ds.tdp.dropna('time', how = 'all')  # drop times with all nan sounding values
    tp = ds.temp.dropna('time', how = 'all')

    if tdp.time.size == 0 or tp.time.size == 0:
        ds = None
        return 
    

    time_valid = np.intersect1d(tdp.time, tp.time)
    ds = ds.where(ds.time.isin(time_valid), drop = True)
    return ds
        

def compute_thermo(ds, pbl_thickness_ratio, pmid):

    """
    Compute specific humidity and sat. specific humidity
    and then layer averaged thetae variables.
    """

    # convert surf. pressure to Pa.
    ds = ds.assign(ps = ds.ps * 100.)

    # compute sp.humidity and sat. sp. humidity values
    ds = ds.assign(hus = qs_calc(ds.p, ds.tdp), 
                                qsat = qs_calc(ds.p, ds.temp))

    # compute theta_e and theta_e_sat
    ds = ds.assign(thetae = theta_e_calc(ds.p, ds.temp, ds.hus),
                   thetae_sat = theta_e_calc(ds.p, ds.temp, ds.qsat))
    
    # Find highest pressure level with valid value    
    p_near_surf = (ds.ps-ds.thetae/ds.thetae).idxmax('p')  # treat this as lowest level (proxy for surf. pressure)
    lower_trop_thickness = p_near_surf - pmid  # find combined thickness of lower troposphere (from near surface to pmid)
    pbl_top = p_near_surf - lower_trop_thickness * pbl_thickness_ratio  # find pbl_top value
    
    thetae_bl, pbl_thickness = _bl_ave(ds.thetae, pbl_top, p_near_surf)
    thetae_lft, lft_thickness = _lft_ave(ds.thetae, pbl_top, pmid)
    thetae_sat_lft, _ = _lft_ave(ds.thetae_sat, pbl_top, pmid)

    wb = (pbl_thickness/lft_thickness) * np.log((pbl_thickness+lft_thickness)/pbl_thickness)

    ds = ds.assign(thetae_bl = thetae_bl, 
                   thetae_lft = thetae_lft, 
                   thetae_sat_lft = thetae_sat_lft, 
                   wb = wb)

    return ds        

def _bl_ave(var, pbl_top:xr.DataArray, 
            pbl_bottom:xr.DataArray)->[xr.DataArray,xr.DataArray]:

    """Returns average over the boundary layer and boundary layer thickness

    Arguments:
    var: variable to be averaged over the boundary layer
    pbl_top: top of the boundary layer in hPa
    pbl_bottom: near-surface pressure level in hPa
    """

    lev = var.p

    ### get trop. contribution ###
    cond = np.logical_and(lev >= pbl_top, lev <= pbl_bottom)
    var_bl = var.where(cond)
    var_bl_contribution = _perform_layer_ave(var_bl, lev)

    pbl_top_lev = xr.where(np.isfinite(var_bl), lev, np.nan).idxmin('p')  #find closest pressure level to pbl_top
    pbl_thickness = pbl_bottom - pbl_top_lev  # compute pbl thickness in hPa

    return var_bl_contribution / pbl_thickness, pbl_thickness
    
def _lft_ave(var, pbl_top:xr.DataArray, 
             pmid:xr.DataArray)->[xr.DataArray,xr.DataArray]:

    """Returns average over the lower-free troposphere and lower-tropospheric thickness

    Arguments:
    var: variable to be averaged over the lower-free troposphere
    pbl_top: top of the boundary layer in hPa
    pmid: mid-level upto which to carry out integration (~ 500 hPa)
    """


    lev = var.p
    cond = np.logical_and(lev >= pmid, lev <= pbl_top)
    var_lft = var.where(cond)
    lft_top_lev = xr.where(np.isfinite(var_lft), lev, np.nan).idxmin('p')  #find closest pressure level to pmid
    pbl_top_lev = xr.where(np.isfinite(var_lft), lev, np.nan).idxmax('p')  #find closest pressure level to pbl_top
    lft_thickness = pbl_top_lev - lft_top_lev
    var_lft_contribution = _perform_layer_ave(var_lft, lev)

    return var_lft_contribution / lft_thickness, lft_thickness


def _perform_layer_ave(var, lev):

    lev_dict = {'p': np.arange(0,lev.size-1)}
    dp = abs(lev.diff(dim = 'p')).assign_coords(lev_dict)
    var1 = var.isel(p = slice(0,lev.size-1)).assign_coords(lev_dict)
    var2 = var.isel(p = slice(1,lev.size)).assign_coords(lev_dict)
    return ((var1+var2) * dp * 0.5).sum('p')

def _get_filenames(path, ext):
    
    """
    For a given pathlib.Path object, search all files
    with given extensions in ext
    """

    fils = []  # create empty fils list to hold search results
    for e in ext:
        search_fils = [str(i) for i in path.glob(f'*.{e}')]  # search for all files with given extension
        if len(search_fils)>0:  # if search is not empty
            fils += search_fils # append to fils
    
    return fils


class ProcessARM:
    def __init__(self, path, var_dict, nprocs = 2,
                 pbl_thickness_ratio = 0.2,  # make pbl thickness 20% of the thickness between surface and pmid.
                 pmid = 500.) -> None:  # mid-level pressure upto which we integrate
        
        self.path = path
        self.nprocs = nprocs  # no of processors to be used for parallel programming
        ext = ['cdf', 'nc'] # list of file extensions to search
        self.files = _get_filenames(path, ext)
        self.files.sort()
        
        self.ds = None
        self.var_dict = var_dict
        self.pbl_thickness_ratio = pbl_thickness_ratio  # in hPa
        self.pmid = pmid  # mid-tropospheric pressure in hPa.

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
    def compute_buoy(cape, subsat, wb):
        return (wb * cape - (1-wb) * subsat) * 9.8/340.0


    def main(self):
        pool = multiprocessing.Pool(self.nprocs)
        result = pool.starmap(read_compute_thermo, 
                              zip(self.files, 
                                  repeat(self.var_dict), 
                                  repeat(self.pbl_thickness_ratio),
                                  repeat(self.pmid)))
        self.result = result
        result = [i for i in result if i is not None]
        if not result:
            raise ValueError("No valid values found in file")
        ds = xr.concat(result, dim = 'time')
        wb = ds.wb  # relative weighting of the boundary layer
        self.cape = self.compute_cape(ds)
        self.subsat = self.compute_subsat(ds)
        self.buoy = self.compute_buoy(self.cape, self.subsat, wb)
        self.precip = ds.prc







