import xarray as xr
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator



# Create bins for plotting
pcp_bins = 2**(np.arange(-2.,7.75,0.125))
pcp_bins = np.insert(pcp_bins,0,1e-3)
pcp_bins = np.insert(pcp_bins,0,0)
pcp_bin_center = (pcp_bins[1:]+pcp_bins[:-1])*0.5

subsat_bins = np.arange(0., 32., 1.5)
instab_bins = np.arange(-40., 16, 1.5)
buoy_bins = np.arange(-20.,5.,1.) * 9.8/340.

buoy_bin_center=(buoy_bins[1:]+buoy_bins[:-1])*0.5

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

class BuoyPrecip:
    
    def __init__(self, precip, buoy, cape, subsat) -> None:
        
        self.precip = precip
        self.buoy = buoy
        self.cape = cape
        self.subsat = subsat

    def __merge(self, ds1, ds2):
        v = np.concatenate((ds1.values, ds2.values))
        return xr.DataArray(v, coords={'time': np.arange(v.size)}, dims=["time"])

    def __add__(self, other):
        """
        dunder magic to pool data together
        """
        if isinstance(other, BuoyPrecip):

            new_precip = self.__merge(self.precip, other.precip)
            new_buoy = self.__merge(self.buoy, other.buoy)
            new_cape = self.__merge(self.cape, other.cape)
            new_sub = self.__merge(self.subsat, other.subsat)

            return BuoyPrecip(new_precip, new_buoy, new_cape, new_sub)
        else:
            raise TypeError("Unsupported operand type. You can only add two BuoyPrecip objects together.")

    # binning

    @staticmethod
    @jit(nopython=True)
    def __bin_1d(x,y,xbins):
        ybinned = np.zeros(xbins.size)
        xhist = np.zeros(xbins.size)
        ystd = np.zeros(xbins.size)
        dx = np.abs(np.diff(xbins)[0])
        xind = ((x-xbins[0])/dx).astype('int')

        for i in np.arange(xbins.size):
            indx = np.where(xind==i)[0]
            ybinned[i] += np.nansum(y[indx])
            xhist[i] = xhist[i]+indx.size
            ystd[i] = np.nanstd(y[indx])
        
        return ybinned, xhist, ystd
    
    def bin_precip_1D(self, buoy_bins = buoy_bins):
        self.pcp_binned_1D, self.buoy_hist_1D, self.pcp_binned_std = self.__bin_1d(self.buoy.values, self.precip.values, 
                            buoy_bins)
        
    def plot_precip_buoy(self, ax, buoy_bins = buoy_bins, **plot_params):
        
        conditional_mean =  np.ma.masked_where(self.buoy_hist_1D<20, 
                                               self.pcp_binned_1D/self.buoy_hist_1D)
        std_err = self.pcp_binned_std/np.sqrt(self.buoy_hist_1D)

        ax.errorbar(buoy_bins, conditional_mean, std_err, fmt = 'o', c = plot_params['color'])
        ax.set_xlabel(plot_params['xlabel'], fontsize = 11)
        ax.set_ylabel(plot_params['ylabel'], fontsize = 11)

    def plot_buoy_hist(self, ax, buoy_bins = buoy_bins, **plot_params):

        dx = abs(np.diff(buoy_bins))[0]
        pdf = self.buoy_hist_1D/(self.buoy_hist_1D.sum() * dx)

        ax.plot(buoy_bins, pdf, color = plot_params['color'], marker = 'o')
        ax.set_xlabel(plot_params['xlabel'], fontsize = 11)


    @staticmethod
    @jit(nopython=True)
    def __bin_2d(x,y,z,xbins,ybins):
        zbinned = np.zeros((xbins.size,ybins.size))
        xy_hist = np.zeros((xbins.size,ybins.size))

        dx = np.abs(np.diff(xbins)[0])
        dy = np.abs(np.diff(ybins)[0])

        xind = ((x-xbins[0])/dx).astype('int')
        yind = ((y-ybins[0])/dy).astype('int')

        for i in np.arange(xbins.size):
            for j in np.arange(ybins.size):
                cond = np.logical_and(xind==i, yind==j)
                indxy = np.where(cond)[0]
                zbinned[i,j] += np.nansum(z[indxy])
                xy_hist[i,j] += indxy.size
        return zbinned, xy_hist


    def bin_precip_2D(self, instab_bins = instab_bins,
                      subsat_bins = subsat_bins):
        
        x = self.cape.values
        y = self.subsat.values
        cond = np.logical_and(np.isfinite(x), np.isfinite(y))
        x = x[cond]
        y = y[cond]
        z = self.precip.values[cond]

        ret=self.__bin_2d(x, y, z, instab_bins, subsat_bins)
        
        self.precip_binned_2D = ret[0]
        self.instab_subsat_hist_2D = ret[1]

    def plot_precip_cape_subsat(self, ax, instab_bins = instab_bins,
                      subsat_bins = subsat_bins, **plot_params):
        
        # set colorbar properties
        new_cmap = truncate_colormap(plot_params['cmap'], 0.1, 0.8)
        levels = MaxNLocator(nbins=15).tick_values(0, 4.)
        norm = BoundaryNorm(levels, ncolors=new_cmap.N, clip=True)

        pr_conditional_mean = np.ma.masked_where(self.instab_subsat_hist_2D == 0, 
                                                 self.precip_binned_2D/self.instab_subsat_hist_2D)


        CT = ax.pcolormesh(subsat_bins, instab_bins, pr_conditional_mean, 
                      cmap = new_cmap, norm = norm)
        # ax.tick_params(which='both', labelsize=11)
        ax.set_xlabel(plot_params['xlabel'], fontsize = 11)
        # ax.set_ylabel(plot_params['ylabel'], fontsize = 11)

        cb = plt.colorbar(CT)
        cb.ax.tick_params(which='both',labelsize=12)
        cb.set_label(label='$\mathrm{mmh^{-1}}$',size=11)
        ax.set_title('Conditional-mean precip.', fontsize = 12)


    def plot_buoy_hist_2D(self, ax, instab_bins = instab_bins,
                      subsat_bins = subsat_bins, **plot_params):

        dx = abs(np.diff(instab_bins))[0]
        dy = abs(np.diff(subsat_bins))[0]

        pdf = self.instab_subsat_hist_2D/(self.instab_subsat_hist_2D.sum() * dx * dy)

        ax.pcolormesh(subsat_bins, instab_bins, pdf, cmap = plot_params['cmap'])
        ax.set_xlabel(plot_params['xlabel'], fontsize = 11)
        ax.set_ylabel(plot_params['ylabel'], fontsize = 11)
        ax.set_title('Joint pdf', fontsize = 12)







