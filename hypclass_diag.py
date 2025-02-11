#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import argparse
import os, sys
import glob
import subprocess
import RIFT.lalsimutils_test as lalsimutils
#import RIFT.lalsimutils as lalsimutils
#import RIFT.lalsimutils_hypclasstest as lalsimutils
import lal
import configparser
config = configparser.ConfigParser(allow_no_value=True)

import time

plt.rcParams.update({
    'font.size': 18,  # Increase font size for all elements
    'axes.titlesize': 22,  # Larger font for titles
    'axes.labelsize': 20,  # Larger font for axis labels
    'xtick.labelsize': 16,  # Larger font for x-axis ticks
    'ytick.labelsize': 16,  # Larger font for y-axis ticks
    'legend.fontsize': 18,  # Larger font for the legend
})

SM = lal.MSUN_SI
lsu_PC=lal.PC_SI
lsu_DimensionlessUnit = lal.DimensionlessUnit


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for finding grid points missing from RIFT composite files.')
    parser.add_argument('--analysis-dir', required=True, type=str, help='Absolute path to the analysis directory.')
    parser.add_argument("--event", type=int, default=0, help='Event number. ADVANCED USERS ONLY')
    parser.add_argument("--split-data", default=False, action='store_true', help='Loads all.net data and splits into hypclass. Must run this once.')
    parser.add_argument("--scatter-plots", action='store_true', help='Plots waveforms for scatter hypclass')
    parser.add_argument("--zw-plots", action='store_true', help='Plots waveforms for ZW hypclass')
    parser.add_argument("--plunge-plots", action='store_true', help='Plots waveforms for plunge hypclass')
    parser.add_argument("--meaningless-plots", action='store_true', help='Plots waveforms for meaningless hypclass')    
    parser.add_argument("--loop-plots", action='store_true', help='enable plot loop')
    return parser.parse_args()

# Parse arguments
args = parse_arguments()


rundir = args.analysis_dir
basedir = os.path.dirname(rundir)

# Create diagnostic directory
diag_path = os.path.join(rundir, 'diagnostic')
if not(os.path.exists(diag_path)):
    os.mkdir(diag_path)
    
# Waveform options not contained in xml
config_file = os.path.join(rundir, 'local.ini')
if not(os.path.exists(config_file)):
    print(f"Config file not found at: {config_file}")
    sys.exit(1)

config.read(config_file)
srate = float(config.get('engine', 'srate'))
deltaT = 1./srate

# PSD info - we're just pulling the H1 PSD here
PSD_path = os.path.join(rundir, 'H1-psd.xml.gz')
channel = 'H1:FAKE-STRAIN'
#deltaF = lalsimutils.get_psd_series_from_xmldoc(PSD_path,'H1').deltaF

# harmonic setting
lmax = int(config.get('rift-pseudo-pipe', 'l-max'))

# segment length
seglen = int(config.get('engine','seglen'))

# dumb hacky way to set deltaF
if seglen == 16:
    deltaF = 0.0625
elif seglen == 32:
    deltaF = 0.031250
elif seglen == 64:
    deltaF = 0.015625
elif seglen == 128:
    deltaF = 0.0078125

#
#
# load the all.net file 
# for each line, take intrinsics and convert to P object
#
# do we have to pull any settings from the config or elsewhere?
#
#run extract_param('hypclass')
#
#
# split results into different bins per hypclass
#
# need to save WFlen, lnL values
#
# need to also save 'meaningless' intrinsic points to give to Jake et al.
#
# We'll want to plot a few also
#
# we also want to be able to impose restrictions by slicing data. For example, look at only 
# points that are close to q = 1, or whatever
#
#

#

if args.split_data:

    # load all.net file
    comp_file = os.path.join(rundir, 'all.net')

    comp_data = np.loadtxt(comp_file)

    m1 = comp_data[:,1]
    m2 = comp_data[:,2]
    s1x = comp_data[:,3]
    s1y = comp_data[:,4]
    s1z = comp_data[:,5]
    s2x = comp_data[:,6]
    s2y = comp_data[:,7]
    s2z = comp_data[:,8]
    E0 = comp_data[:,9]
    p_phi0 = comp_data[:,10]
    lnL = comp_data[:,11]
    sigma = comp_data[:,12]
    ntotal = comp_data[:13]

    n_samples = len(m1)

    print(f'Total of {n_samples} samples in the all.net file')

    # initialize
    #P = lalsimutils.ChooseWaveformParams()

    # These are the injected parameters
    if not(args.event == 0):
        truth_xml = os.path.join(basedir, f'mdc_{args.event}.xml.gz')
    else:
        truth_xml = os.path.join(basedir, 'mdc.xml.gz')

    P_truth = lalsimutils.xml_to_ChooseWaveformParams_array(truth_xml)[0] # taking the settings from the injection. We'll override the intrinsics only.

    hypclass_list = []
    #WFlen_list = []

    # Initialize lists to hold data for each hypclass category
    scatter_data = []
    plunge_data = []
    zoomwhirl_data = []
    meaningless_data = []
    unknown_data = []

    #
    eval_start = time.time()

    for i in range(n_samples):
    #for i in range(2):
        P = P_truth.copy()
        # intrinsics
        P.m1 = m1[i]*SM
        P.m2 = m2[i]*SM
        P.s1x = s1x[i]
        P.s1y = s1y[i]
        P.s1z = s1z[i]
        P.s2x = s2x[i]
        P.s2y = s2y[i]
        P.s2z = s2z[i]
        P.E0 = E0[i]
        P.p_phi0 = p_phi0[i]

        # other
        P.deltaF = deltaF
        P.deltaT = deltaT

        #P.print_params()


        #hypclass, WFlen = P.extract_param('hypclass')
        hypclass = P.extract_param('hypclass')
        hypclass_list.append(hypclass)
        #WFlen_list.append(WFlen)

        # Append the current line of comp_data to the corresponding category list
        if hypclass == 'scatter':
            scatter_data.append(comp_data[i])
        elif hypclass == 'plunge':
            plunge_data.append(comp_data[i])
        elif hypclass == 'zoomwhirl':
            zoomwhirl_data.append(comp_data[i])
        elif hypclass == 'meaningless':
            meaningless_data.append(comp_data[i])
        elif hypclass == 'unknown':
            unknown_data.append(comp_data[i])

    eval_end = time.time()

    total_eval_time = eval_end - eval_start

    #print(hypclass_list)
    #print(WFlen_list)

    print(f'Total number of scatters is {hypclass_list.count("scatter")} out of {n_samples} total points.')
    print(f'Total number of plunges is {hypclass_list.count("plunge")} out of {n_samples} total points.')
    print(f'Total number of zoomwhirls is {hypclass_list.count("zoomwhirl")} out of {n_samples} total points.')
    print(f'Total number of meaningless is {hypclass_list.count("meaningless")} out of {n_samples} total points.')
    print(f'Total number of unknown is {hypclass_list.count("unknown")} out of {n_samples} total points.')


    #if np.amax(np.abs(np.diff(np.array(WFlen_list)))) > 0.0:
    #    print('Difference in WFlen detected!')
        #print(WFlen_list)
    #    longest_len = np.amax(np.array(WFlen_list))
    #    ll_indx = np.argmax(np.array(WFlen_list))
    #    shortest_len = np.amin(np.array(WFlen_list))
    #    sl_indx = np.argmin(np.array(WFlen_list))
    #    print(f'Largest WFlen is {longest_len}, corresponding to {longest_len*deltaT} seconds. Corresponding hypclass is {hypclass_list[ll_indx]}.')
    #    print(f'Shortest WFlen is {shortest_len}, corresponding to {shortest_len*deltaT} seconds. Corresponding hypclass is {hypclass_list[sl_indx]}.')
    #else:
    #    WFlen_time = WFlen_list[0]*deltaT
    #    print(f'WFlen is consistently {WFlen_list[0]} for all waveforms. This corresponds to {WFlen_time} seconds.')

    print(f'Note: total hypclass evaluation time was {total_eval_time} seconds for {n_samples} waveforms.')

    # Dump the categorized data to text files
    np.savetxt(os.path.join(diag_path, 'scatter_hypclass-data.txt'), scatter_data)
    np.savetxt(os.path.join(diag_path, 'plunge_hypclass-data.txt'), plunge_data)
    np.savetxt(os.path.join(diag_path, 'zoomwhirl_hypclass-data.txt'), zoomwhirl_data)
    np.savetxt(os.path.join(diag_path, 'meaningless_hypclass-data.txt'), meaningless_data)
    np.savetxt(os.path.join(diag_path, 'unknown_hypclass-data.txt'), unknown_data)
    
elif args.loop_plots:
    # load true params from mdc.xml.gz - only need to do this once
    # loop through each hypclass-data.txt file
    # for each one, load the file like an all.net file
    # for each row, convert to P object (as a modified duplicate of the true mdc.xml)
    # run wav_gen - output h(t), hlm(t), possibly h(f), hlm(f)
    
    
    ##############################
    # Function definitions
    ##############################

    def read_frame(frame_file, channel):
        frame_data = {}
        #raw_data = lalsimutils.frame_data_to_hoft(frame_file, channel)
        raw_data = TimeSeries.read(frame_file, channel)
        frame_data['strain'] = np.array(raw_data.value)
        frame_data['sample_times'] = np.array(raw_data.times)

        return frame_data

    k_to_lm = {
        "0": (2, 1), "-0": (2, -1), "20": (2, 0), "1": (2, 2), "-1": (2, -2),
        "2": (3, 1), "-2": (3, -1), "3": (3, 2), "-3": (3, -2), "4": (3, 3),
        "-4": (3, -3), "30": (3, 0), "5": (4, 1), "-5": (4, -1), "6": (4, 2),
        "-6": (4, -2), "7": (4, 3), "-7": (4, -3), "8": (4, 4), "-8": (4, -4),
        "40": (4, 0)
    }

    def get_lm_from_k(k):
        return k_to_lm.get(str(k))


    def gen_wav(P):
        # stripped down version of gen_wav from the inject_compare_ILE.py code.
        # currently just returns h(t), hlmoft

        # create h(t) and hlm(t) from hlmoff - this actually runs hlmoft then FFTs it
        hlmoft = {}
        hlmoft_times = {}
        hlmoff, hlmoff_conj = lalsimutils.std_and_conj_hlmoff(P, Lmax=lmax)
        for mode in hlmoff:
            hlmoft[mode] = lalsimutils.DataInverseFourier(hlmoff[mode]) # convert to time
            hlmoft_times[mode] = lalsimutils.evaluate_tvals(hlmoft[mode])

        hoft = lalsimutils.hoft_from_hlm(hlmoft, P, return_complex=False) # combine modes
        
        hoft_times = lalsimutils.evaluate_tvals(hoft)

        return hoft, hoft_times, hlmoft, hlmoft_times
    
    def load_data_to_P(data_path, P_truth):
        
        comp_data = np.loadtxt(data_path)
        
        m1 = comp_data[:,1]
        m2 = comp_data[:,2]
        s1x = comp_data[:,3]
        s1y = comp_data[:,4]
        s1z = comp_data[:,5]
        s2x = comp_data[:,6]
        s2y = comp_data[:,7]
        s2z = comp_data[:,8]
        E0 = comp_data[:,9]
        p_phi0 = comp_data[:,10]
        lnL = comp_data[:,11]
        sigma = comp_data[:,12]
        ntotal = comp_data[:13]

        n_samples = len(m1)
        
        P_list = []
        for i in range(n_samples):
        #for i in range(2):
            P = P_truth.copy()
            # intrinsics
            P.m1 = m1[i]*SM
            P.m2 = m2[i]*SM
            P.s1x = s1x[i]
            P.s1y = s1y[i]
            P.s1z = s1z[i]
            P.s2x = s2x[i]
            P.s2y = s2y[i]
            P.s2z = s2z[i]
            P.E0 = E0[i]
            P.p_phi0 = p_phi0[i]

            # other
            P.deltaF = deltaF
            P.deltaT = deltaT
            
            P_list.append(P)
        
        return P_list, lnL
        






    ##############################
    # Injected Signal
    ##############################

    # These are the injected parameters
    if not(args.event == 0):
        truth_xml = os.path.join(basedir, f'mdc_{args.event}.xml.gz')
    else:
        truth_xml = os.path.join(basedir, 'mdc.xml.gz')


    P_truth = lalsimutils.xml_to_ChooseWaveformParams_array(truth_xml)[0] # taking the settings from the injection. We'll override the intrinsics only.   

    P_truth.detector = 'H1'
    P_truth.deltaT = 1./srate
    P_truth.deltaF = deltaF
    P_truth.radec = True

    hoft_truth, hoft_truth_times, hlmoft_truth, hlmoft_truth_times = gen_wav(P_truth)


    #print(dir(hoft_truth))
    #print(hoft_truth.epoch)
    
    #print(hlmoft
    
    
    
    
    
    ##############################
    # Plotting loop
    ##############################    
    
    # Loading from diagnostic    
    if args.scatter_plots:        
        data_path = os.path.join(diag_path, 'scatter_hypclass-data.txt')
        name_label = 'Scatter'
    elif args.zw_plots:        
        data_path = os.path.join(diag_path, 'zoomwhirl_hypclass-data.txt')
        name_label = 'Zoom-whirl'
    elif args.plunge_plots:        
        data_path = os.path.join(diag_path, 'plunge_hypclass-data.txt')
        name_label = 'Plunge'
    elif args.meaningless_plots:        
        data_path = os.path.join(diag_path, 'meaningless_hypclass-data.txt')
        name_label = 'Meaningless'
        
    imagepath_base = f'/home/chad.henshaw/public_html/hyperbolic_waveforms/hypclass_diag_images_021025/{name_label}/'
    
    if not(os.path.exists(imagepath_base)):
        os.mkdir(imagepath_base)
    
    Plist, lnL = load_data_to_P(data_path, P_truth)
    
    for index, val in enumerate(Plist):
        
        fig, axs = plt.subplots(figsize=(20, 10), nrows=1, ncols=1)
        #ax = axs.flatten()
        
        # always plotting the injected params to set a default plot scale
        axs.plot(hoft_truth_times, hoft_truth.data.data, color = 'blue', label='Injected params', alpha=0.6)
        
        
        print(f'Evaluating index {index}')
        val.print_params()
        
        hoft, hoft_times, hlmoft, hlmoft_times = gen_wav(val) # each val is a P object
        
        
        
        
        
        
        axs.plot(hoft_times, hoft.data.data, color ='orange', linestyle ='dashed', label=f'{name_label}-{index}')
        
        
    
    
    
    
        axs.set_xlabel('Time [s]')
        axs.set_ylabel('Strain') 

        # I think we want to allow the natural dynamic limit selection

        axs.grid(alpha=0.3)
        axs.legend()
        
        textstr_inj = '\n'.join((
                'Injected (blue):',
                '$M = %.1f \: [M_{\odot}]$' % ((P_truth.m1 + P_truth.m2)/SM),
                '$q = %.3f $' % (P_truth.m1/P_truth.m2),
                '$S_{1,z}=%.2f$' % (P_truth.s1z),
                '$S_{2,z}=%.2f$' % (P_truth.s2z),
                '$E_0=%.5f$' % (P_truth.E0),
                '$p_{\phi}^0=%.5f$' % (P_truth.p_phi0)))

        axs.text((1.0-0.0200), 0.02, textstr_inj, transform=axs.transAxes, fontsize=12, horizontalalignment='right', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        textstr_val = '\n'.join((
                'Sample (orange):',
                '$M = %.1f \: [M_{\odot}]$' % ((val.m1 + val.m2)/SM),
                '$q = %.3f $' % (val.m1/val.m2),
                '$S_{1,z}=%.2f$' % (val.s1z),
                '$S_{2,z}=%.2f$' % (val.s2z),
                '$E_0=%.5f$' % (val.E0),
                '$p_{\phi}^0=%.5f$' % (val.p_phi0),
                '$\ln{L}=%.3f$' % (lnL[index])))

        axs.text((1.0-0.2200), 0.02, textstr_val, transform=axs.transAxes, fontsize=12, horizontalalignment='right', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        plt.tight_layout()    
        
        savename = f'{name_label}-{index}_mtot-{(val.m1 + val.m2)/SM}_q-{val.m1/val.m2}_chi1-{val.s1z}_chi2-{val.s2z}_E0-{val.E0}_pphi0-{val.p_phi0}_lnL-{lnL[index]}.png'        
        savepath = os.path.join(imagepath_base, savename)
        
        plt.savefig(savepath, bbox_inches='tight')
        
        plt.close(fig)
    
    
    
    
# next
# submit to condor