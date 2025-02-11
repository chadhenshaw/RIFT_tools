#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import argparse
import os, sys
import glob
import subprocess
import RIFT.lalsimutils as lalsimutils
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
    parser.add_argument('--user-test', action='store_true', help='Runs the user test params')
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
if not os.path.exists(PSD_path):  # Check if the file exists
    PSD_path = '/home/chad.henshaw/Injections/PSDs/O4/H1-psd.xml.gz'
    print(f'No H1 PSD in rundir, loading from {PSD_path}')
channel = 'H1:FAKE-STRAIN'
PSD_path_CE = '/home/chad.henshaw/Injections/PSDs/CE/C1_at_H1_fromascii_psd.xml.gz'
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
    
LIGO_PSD = lalsimutils.get_psd_series_from_xmldoc(PSD_path, 'H1')
LIGO_freqs = np.array([LIGO_PSD.f0 + i * LIGO_PSD.deltaF for i in range(LIGO_PSD.data.length)])
LIGO_ASD = np.sqrt(np.array([LIGO_PSD.data.data[i] for i in range(LIGO_PSD.data.length)]))

CE_PSD = lalsimutils.get_psd_series_from_xmldoc(PSD_path_CE, 'H1')
CE_freqs = np.array([CE_PSD.f0 + i * CE_PSD.deltaF for i in range(CE_PSD.data.length)])
CE_ASD = np.sqrt(np.array([CE_PSD.data.data[i] for i in range(CE_PSD.data.length)]))

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


def gen_wav_old(P):
    # stripped down version of gen_wav from the inject_compare_ILE.py code.
    # currently just returns h(t), hlmoft

    # create h(t) and hlm(t) from hlmoff - this actually runs hlmoft then FFTs it
    hlmoft = {}
    hlmoft_times = {}
    hlmoff_freqs = {}
    hlmoff, hlmoff_conj = lalsimutils.std_and_conj_hlmoff(P, Lmax=lmax)
    for mode in hlmoff:
        hlmoft[mode] = lalsimutils.DataInverseFourier(hlmoff[mode]) # convert to time
        hlmoft_times[mode] = lalsimutils.evaluate_tvals(hlmoft[mode]) + 1e9
        #hlmoff_freqs[mode] = lalsimutils.evaluate_fvals(hlmoff[mode])
        
        print(f'gen wav mode {mode}')
        print(hlmoff[mode].f0)
        print(hlmoff[mode].deltaF)
        print(hlmoff[mode].data.length)
        
        
        hlmoff_freqs[mode] = np.array([hlmoff[mode].f0 + j*hlmoff[mode].deltaF for j in range(hlmoff[mode].data.length)])

    hoft = lalsimutils.hoft_from_hlm(hlmoft, P, return_complex=False) # combine modes
    #hoff = lalsimutils.DataFourier(hoft)

    hoft_times = lalsimutils.evaluate_tvals(hoft)
    #hoff_freqs = lalsimutils.evaluate_fvals(hoff)

    return hoft, hoft_times, hlmoft, hlmoft_times, hlmoff, hlmoff_freqs

def gen_wav(P):
    # Going to duplicate the action of std_and_conj_hlmoff
    print('Generating hlmoft with gen_wav')
    hlms = lalsimutils.hlmoft(P, Lmax=lmax)
    hlmsT = hlms # copying the time domain hlms
    
    if isinstance(hlms, dict):
        hlmsF = {}
        hlms_conj_F = {}
        hlmsT_times = {}
        for mode in hlms:
            print(f'Evaluating frequency domain of mode: {mode}')
            print(f'TDlen: {hlms[mode].data.length}')
            print(f'epoch: {hlms[mode].epoch}')
            print(f'f0: {hlms[mode].f0}')
            
            hlmsT_times[mode] = lalsimutils.evaluate_tvals(hlmsT[mode]) + 1e9
            
            
            hlmsF[mode] = lalsimutils.DataFourier(hlms[mode])
            hlms[mode].data.data = np.conj(hlms[mode].data.data)
            hlms_conj_F[mode] = lalsimutils.DataFourier(hlms[mode])
            
        #returns hlmsF, hlms_conj_F as dicts
        
    # get hoft
    
    hoft = lalsimutils.hoft_from_hlm(hlmsT, P, return_complex=False)
    hoft_times = lalsimutils.evaluate_tvals(hoft)
    
    # trying alternate hoff
    
    hoff = lalsimutils.DataFourierREAL8(hoft)
    #hoff_freqs = lalsimutils.evaluate_fvals(hoff)
    print(f'hoff.f0 is {hoff.f0}')
    hoff_freqs = np.array([hoff.f0 + i * hoff.deltaF for i in range(hoff.data.length)])
        
    # now to get time and frequency stamps
    
    return hlmsT, hlmsT_times, hlmsF, hoft, hoft_times, hoff, hoff_freqs
    


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

hlmoft_truth, hlmoft_truth_times, hlmoff_truth, hoft_truth, hoft_truth_times, hoff_truth, hoff_truth_freqs = gen_wav(P_truth)

#for mode in hlmoff_truth.keys():
#    print(mode)
#    print(hlmoff_truth[mode].data.data)




#sys.exit()

#hoft_truth, hoft_truth_times, hlmoft_truth, hlmoft_truth_times, hlmoff_truth, hlmoff_truth_freqs = gen_wav(P_truth)

#print(hlmoff_truth_freqs)
#sys.exit()

#print(hlmoff_truth[(2,2)].data.data)
#print(dir(hlmoff_truth[(2,2)]))

#sys.exit()

#hoft, hoft_times, hlmoft, hlmoft_times, hoff, hoff_freqs, hlmoff, hlmoff_freqs

#print(hoff)
#print(hoff_freqs)


##############################
# User Test
##############################

if args.user_test:
    P_test = P_truth.copy()
    P_test.m1 = 61.6107*SM
    P_test.m2 = 33.8893*SM
    P_test.s1x = 0.0
    P_test.s1y = 0.0
    P_test.s1z = 0.0
    P_test.s2x = 0.0
    P_test.s2y = 0.0
    P_test.s2z = 0.0
    P_test.E0 = 1.06597
    P_test.p_phi0 =  7.82595

    #hoft_test, hoft_test_times, hlmoft_test, hlmoft_test_times = gen_wav(P_test)
    #hlmoft_test, hlmoft_test_times, hlmoff_test = gen_wav(P_test)
    
    hlmoft_test, hlmoft_test_times, hlmoff_test, hoft_test, hoft_test_times, hoff_test, hoff_test_freqs = gen_wav(P_test)


##############################
# Plotting
##############################    

truth_mode_styles = {
    (2, 2): {'color': 'blue'},
    (2, -2): {'color': 'deepskyblue'},
    (2, 1): {'color': 'green'},
    (2, -1): {'color': 'limegreen'},
    (2, 0): {'color': 'purple'},

    (3, 3): {'color': 'darkmagenta'},
    (3, -3): {'color': 'orchid'},
    (3, 2): {'color': 'darkcyan'},
    (3, -2): {'color': 'cadetblue'},
    (3, 1): {'color': 'goldenrod'},
    (3, -1): {'color': 'gold'},
    (3, 0): {'color': 'crimson'},

    (4, 4): {'color': 'darkred'},
    (4, -4): {'color': 'firebrick'},
    (4, 3): {'color': 'darkblue'},
    (4, -3): {'color': 'royalblue'},
    (4, 2): {'color': 'darkorange'},
    (4, -2): {'color': 'chocolate'},
    (4, 1): {'color': 'olive'},
    (4, -1): {'color': 'yellowgreen'},
    (4, 0): {'color': 'indigo'}
}

test_mode_styles = {
    (2, 2): {'color': 'red'},
    (2, -2): {'color': 'tomato'},
    (2, 1): {'color': 'orange'},
    (2, -1): {'color': 'darkorange'},
    (2, 0): {'color': 'brown'},

    (3, 3): {'color': 'darkviolet'},
    (3, -3): {'color': 'plum'},
    (3, 2): {'color': 'teal'},
    (3, -2): {'color': 'lightseagreen'},
    (3, 1): {'color': 'darkgoldenrod'},
    (3, -1): {'color': 'khaki'},
    (3, 0): {'color': 'maroon'},

    (4, 4): {'color': 'darkslategray'},
    (4, -4): {'color': 'black'},
    (4, 3): {'color': 'navy'},
    (4, -3): {'color': 'steelblue'},
    (4, 2): {'color': 'orangered'},
    (4, -2): {'color': 'sienna'},
    (4, 1): {'color': 'darkolivegreen'},
    (4, -1): {'color': 'chartreuse'},
    (4, 0): {'color': 'midnightblue'}
}


default_truth_style = {'color': 'black'}  # Fallback for unknown modes
default_test_style = {'color': 'gray'}   # Fallback for unknown modes

linestyles = ['solid', 'dashed', 'dotted']  # Cycling linestyles




fig, axs = plt.subplots(figsize=(20, 10), nrows=2, ncols=1)
ax = axs.flatten()

# always plotting the injected params full WF
#axs.plot(hoft_truth_times, hoft_truth.data.data, color = 'blue', label='Injected params - combined', alpha=0.6)

#axs.plot(hoft_test_times, hoft_test.data.data, color = 'green', label='test params - combined', alpha=0.6)

for i, mode in enumerate(hlmoft_truth.keys()):      
    if not(np.amax(hlmoft_truth[mode].data.data) == 0j):
        print(f'Evaluating {mode} for true params')
        
        style = truth_mode_styles.get(mode, default_truth_style)  # Color from truth dict
        linestyle = linestyles[i % len(linestyles)] # Cycle through linestyles
        
        ax[0].plot(hlmoft_truth_times[mode], np.abs(hlmoft_truth[mode].data.data), linestyle=linestyle, label = f'{mode} - true params, amp', **style)
        #axs.plot(hlmoft_truth_times[mode], np.real(hlmoft_truth[mode].data.data), linestyle='dashed', label = f'{mode} - true params, real', **style)
        #axs.plot(hlmoft_truth_times[mode], np.imag(hlmoft_truth[mode].data.data), linestyle='dotted', label = f'{mode} - true params, imag', **style)
        
        #ax[1].plot(hlmoff_truth_freqs[mode], np.abs(hlmoff_truth[mode].data.data), linestyle=linestyle, **style)
        
        #ax[1].plot(np.abs(hlmoff_truth[mode].data.data), linestyle=linestyle, **style)

ax[1].loglog(hoff_truth_freqs, np.real(hoff_truth.data.data), linestyle='solid', label=f'truth hoff from hoft', color='blue')
        #ax[1].plot(np.real(hlmoff_truth[mode].data.data), linestyle=linestyle, **style, label='real')
        
        
            
if args.user_test:
    for i, mode in enumerate(hlmoft_test.keys()):
        if not(np.amax(hlmoft_test[mode].data.data) == 0j):
            print(f'Evaluating {mode} for test params')
            
            style = test_mode_styles.get(mode, default_test_style)  # Color from test dict
            linestyle = linestyles[i % len(linestyles)] # Cycle through linestyles
            
            ax[0].plot(hlmoft_test_times[mode], np.abs(hlmoft_test[mode].data.data), linestyle=linestyle, label = f'{mode} - test params, amp', **style)
            #axs.plot(hlmoft_test_times[mode], np.real(hlmoft_test[mode].data.data), linestyle='dashed', label = f'{mode} - test params, real')
            #axs.plot(hlmoft_test_times[mode], np.imag(hlmoft_test[mode].data.data), linestyle='dotted', label = f'{mode} - test params, imag')
            
    ax[1].loglog(hoff_test_freqs, np.real(hoff_test.data.data), linestyle='dashed', label=f'test hoff from hoft', alpha=0.3, color='orange')
        
            
ax[1].loglog(LIGO_freqs, LIGO_ASD, label=r'aLIGO O4 design: $1/\sqrt{Hz}$', color='black')
ax[1].loglog(CE_freqs, CE_ASD, label=r'CE design: $1/\sqrt{Hz}$', color='black', linestyle='dotted')

ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Strain') 

ax[1].set_xlabel('Freqs [Hz]')
ax[1].set_ylabel('Strain')


ax[0].set_xlim(-12.0 + 1e9, 5.0 + 1e9)
ax[1].set_xlim(1.0, 4095.0)
ax[1].set_ylim(1e-26, 1e-20)
#ax[1].set_xscale('log')

ax[0].grid(alpha=0.3)
ax[1].grid(alpha=0.3)
ax[0].legend(loc='upper left')
ax[1].legend(loc='upper right')

textstr_inj = '\n'.join((
        'True Params:',
        '$M = %.1f \: [M_{\odot}]$' % ((P_truth.m1 + P_truth.m2)/SM),
        '$q = %.3f $' % (P_truth.m1/P_truth.m2),
        '$S_{1,z}=%.2f$' % (P_truth.s1z),
        '$S_{2,z}=%.2f$' % (P_truth.s2z),
        '$E_0=%.5f$' % (P_truth.E0),
        '$p_{\phi}^0=%.5f$' % (P_truth.p_phi0)))

ax[0].text((1.0-0.0200), 0.02, textstr_inj, transform=ax[0].transAxes, fontsize=12, horizontalalignment='right', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

if args.user_test:
    textstr_test = '\n'.join((
            'Test Params:',
            '$M = %.1f \: [M_{\odot}]$' % ((P_test.m1 + P_test.m2)/SM),
            '$q = %.3f $' % (P_test.m1/P_test.m2),
            '$S_{1,z}=%.2f$' % (P_test.s1z),
            '$S_{2,z}=%.2f$' % (P_test.s2z),
            '$E_0=%.5f$' % (P_test.E0),
            '$p_{\phi}^0=%.5f$' % (P_test.p_phi0)))

    ax[0].text((1.0-0.1200), 0.02, textstr_test, transform=ax[0].transAxes, fontsize=12, horizontalalignment='right', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


plt.tight_layout()

imagepath_base = '/home/chad.henshaw/Injections/RIFT/dev/single_inject/hyperbolic/scatter/seglen_tests/taper-update_64seglen_flow-10_mtot-100.0_q-1.0_xi-0.0_E0-1.01_pphi0-4.4_SNR-20/analysis_0_nospin_10kgrid/diagnostic/'

#savename = f'{name_label}-{index}_mtot-{(val.m1 + val.m2)/SM}_q-{val.m1/val.m2}_chi1-{val.s1z}_chi2-{val.s2z}_E0-{val.E0}_pphi0-{val.p_phi0}_lnL-{lnL[index]}.png'

savename = 'mode_test.png'
#savepath = os.path.join(imagepath_base, savename)
savepath = os.path.join(diag_path, savename)

plt.savefig(savepath, bbox_inches='tight')

plt.close(fig)
    
    
    
    
# next
# submit to condor