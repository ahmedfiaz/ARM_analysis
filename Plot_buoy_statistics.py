import matplotlib.pyplot as plt

def plot_bl_stats(ds, plot_params, SAVE_PLOT = False):
    
    fig,axx=plt.subplots(2, 2, figsize = (8,6))

    ax = axx[0,0]
    ds.plot_precip_buoy(ax, **plot_params)
    ax.set_ylim(bottom = 0)

    ax = axx[0, 1]
    ds.plot_buoy_hist(ax, **plot_params)
    ax.set_title('Pdf of $\mathrm{B_L}$', fontsize = 12)

    plot_params.update(cmap = plt.get_cmap('gist_ncar'), 
                       xlabel = '$\mathrm{subsat_L}$', ylabel = '$\mathrm{instab_L}$')
    ax= axx[1,0]
    ds.plot_buoy_hist_2D(ax, **plot_params)


    ax = axx[1,1]
    ds.plot_precip_cape_subsat(ax, **plot_params)

    fig.suptitle('$\mathrm{B_L}$ statistics for ' + plot_params['plt_title'])
    fig.subplots_adjust(top = 0.8)

    plt.tight_layout()

    if SAVE_PLOT:
        plt.savefig(plot_params['plt_dir']+'P_BL_statistics_{}.pdf'.format(plot_params['plt_title']),format='pdf',
                   bbox_inches='tight')