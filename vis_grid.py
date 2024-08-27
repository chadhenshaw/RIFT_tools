#!/usr/bin/env python3

## RIFT grid visualization - for use with single_inject_RIFT.py


#import RIFT.lalsimutils as lalsimutils
import RIFT
import numpy as np
#import precession
#import random
#from tabulate import tabulate
#import matplotlib.pyplot as plt
#from pycbc import pnutils
#from pycbc.waveform import get_td_waveform
#import h5py
#import lal
#import json
#from ligo.lw import lsctables, table, utils
import os
import sys
#import shutil
import glob
from RIFT.misc.dag_utils import mkdir
import argparse
import re


## Argparse ##

parser = argparse.ArgumentParser()
parser.add_argument("--analysis-dir", default=None, type=str, required=True, help="REQUIRED. Path to analysis directory.")
parser.add_argument("--hyperbolic",action='store_true',help="Enable for hyperbolic analyses.")
parser.add_argument("--alt-coord-list", default=None, type=str, help="List of coordinates to plot")
opts =  parser.parse_args()

hyperbolic = False
if opts.hyperbolic:
    hyperbolic=True


# get to analysis directory

ad_path = os.path.abspath(opts.analysis_dir)
meta_path = os.path.dirname(ad_path)

os.chdir(ad_path)
mkdir('vis_plots')

plot_dest = os.path.join(ad_path, 'vis_plots/')

# get truth file:

truth_xml = os.path.join(meta_path, 'mdc.xml.gz')

###
# Functions
###


def load_init_grid_data(grid, dat_path, net_path, hyperbolic=False):
    # first, convert the xml to temporary dat
    

    if hyperbolic:
        cmd = f'convert_output_format_ile2inference --export-hyperbolic {grid} > {dat_path} '
    else:    
        cmd = f'convert_output_format_ile2inference {grid} > {dat_path} '
    os.system(cmd)
    
    # load the dat
    init_data = np.loadtxt(dat_path)    
    m1 = init_data[:,0]
    m2 = init_data[:,1]
    s1x = init_data[:,2]
    s1y = init_data[:,3]
    s1z = init_data[:,4]
    s2x = init_data[:,5]
    s2y = init_data[:,6]
    s2z = init_data[:,7]

    if hyperbolic:
        E0 = init_data[:,24]
        p_phi0 = init_data[:,25]


    if not hyperbolic:

        # create fake columns to match all.net format
        data_len = len(m1)
        new_col_0 = -1*np.ones(data_len)
        new_col_9 = np.random.uniform(10.0, 50.0, data_len)
        new_col_10 = np.random.uniform(0.1, 0.2, data_len)
        new_col_11 = np.random.uniform(100000.0, 500000.0, data_len)
        new_col_12 = -1*np.ones(data_len)
        # total number of cols is 13
        n_cols = 13

        # stack into combined array
        new_array = np.column_stack((new_col_0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, new_col_9, new_col_10, new_col_11, new_col_12))
    
        # Define formatting strings for int and float values
        int_fmt = "%d"    # For integers
        float_fmt = "%.5f"  # For floats
    
        # Create a list of formatting strings for each column
        fmt_list = [int_fmt] + [float_fmt] * (n_cols - 2) + [int_fmt]

    else:
        # create fake columns to match all.net format
        data_len = len(m1)
        new_col_0 = -1*np.ones(data_len)
        new_col_11 = np.random.uniform(10.0, 50.0, data_len)
        new_col_12 = np.random.uniform(0.1, 0.2, data_len)
        new_col_13 = np.random.uniform(100000.0, 500000.0, data_len)
        new_col_14 = -1*np.ones(data_len)
        # total number of cols is 15
        n_cols = 15

        # stack into combined array
        new_array = np.column_stack((new_col_0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, E0, p_phi0, new_col_11, new_col_12, new_col_13, new_col_14))
    
        # Define formatting strings for int and float values
        int_fmt = "%d"    # For integers
        float_fmt = "%.5f"  # For floats
    
        # Create a list of formatting strings for each column
        fmt_list = [int_fmt] + [float_fmt] * (n_cols - 2) + [int_fmt]
    
    
    
    # save output all.net file
    np.savetxt(net_path, new_array, fmt=fmt_list)
    
    return


def ppc(dat_path, net_path, iteration, coord_list):
    os.chdir(plot_dest) 
    
    ppc_cmd = f'plot_posterior_corner.py --posterior-file {dat_path} --composite-file {net_path} --truth-file {truth_xml} --ci-list [1.0] --use-all-composite-but-grayscale --lnL-cut 40.0 '

    for coord in coord_list:
        ppc_cmd += f' --parameter {coord} '


    ppc_cmd += f' --posterior-label "it-{iteration}" --use-legend '
    
    if hyperbolic:
        ppc_cmd += ' --hyperbolic '
    

    print(ppc_cmd)
    os.system(ppc_cmd)
    
    def_fig_name = os.path.join(plot_dest, 'corner_'+'_'.join(coord_list)+'.png')
    os.rename(def_fig_name, os.path.join(plot_dest, f'it{iteration}_corner_'+'_'.join(coord_list)+'.png'))
    
    
#param_postfix = "_".join(opts.parameter)


#load_grid_data(first_grid)


## Define plot coordinates ##

default_intrinsic_coords = ['mc', 'eta', 'chi_eff', 'chi_p']

if opts.alt_coord_list is not None:
    coord_list = opts.alt_coord_list
else:
    coord_list = default_intrinsic_coords

if hyperbolic:
    coord_list = ['mtotal', 'q', 'p_phi0', 'E0', 'chi_eff']
    #coord_list.append('p_phi0')
    #coord_list.append('E0')



#find grids:

search_pattern = os.path.join(ad_path, 'overlap-grid-*.xml.gz')
grid_files = glob.glob(search_pattern)
grid_files.sort(key=lambda x: int(re.search(r'overlap-grid-(\d+)\.xml\.gz', x).group(1))) # sort list

#first_grid = grid_files[0]

# do special stuff for initial grid
print('Visualizing initial grid')
        
init_dat_path = os.path.join(plot_dest, "tmp_grid.dat")
init_net_path = os.path.join(plot_dest, "vis-tmp_all.net")
load_init_grid_data(grid_files[0], init_dat_path, init_net_path, hyperbolic)
ppc(init_dat_path, init_net_path, iteration=0, coord_list=coord_list)


#print('Stopping here')
#sys.exit(0)

# iterate through remaining grids
comp_list = []
for indx, grid in enumerate(grid_files[1:]):
    dat_indx = int(indx + 1)
    print(f'Visualizing iteration {dat_indx}')
    dat_path = os.path.join(ad_path, f'posterior_samples-{dat_indx}.dat')
    
    comp_list.append(os.path.join(ad_path, f'consolidated_{indx}.composite'))
    
    # need to add together comp files
    if len(comp_list) == 1:
        total_comp = comp_list[0]
    
    else:
        # add together
        add_cmd = 'util_CleanILE.py '
        if hyperbolic:
            add_cmd += ' --hyperbolic '
        
        for comp in comp_list:
            add_cmd += f' {comp} '
            
        total_comp = os.path.join(plot_dest, 'tmp_all.composite')
        
        add_cmd += f' > {total_comp} '
        
        print(add_cmd)
        os.system(add_cmd)
        
    net_path = total_comp
    
    ppc(dat_path, net_path, iteration=dat_indx, coord_list=coord_list)
            
    
                            
        
        #net_path = # need to combine composite files






# we now have two files: tmp_grid.dat (samples file) and vis-tmp_all.net (composite file)


# next, we create plots
