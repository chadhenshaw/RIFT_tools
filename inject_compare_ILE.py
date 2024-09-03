#!/usr/bin/env python3

# Comparing likelihood values for an injection

#import RIFT.lalsimutils_test as lalsimutils
import RIFT.lalsimutils as lalsimutils

import os, sys

#import EOBRun_module
import numpy as np
from matplotlib import pyplot as plt
#import time
import lal
#import lalsimulation as lalsim
#import scipy.signal as sp

#import gw_cwt

#import glob
#from PIL import Image
import re
natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

SM = lal.MSUN_SI
lsu_PC=lal.PC_SI

##############################
# Global settings
##############################

lmax = 4
calc_ILE = True
user_test = True
maxL = True
morph = 'spin-scatter' # set to capture or scatter
#evaluate_cwt = True # set to true to run CWT on the injected signal (from hlmoft reconstruction)

# for plotting
if morph == 'capture':
    plot_domain = [-1.0, 0.25]
elif morph == 'scatter' or morph == 'spin-scatter':
    plot_domain = [-0.1, 0.1]
elif morph == 'capture2':
    plot_domain = [-1.0, 0.25]
elif morph == 'capturetest':
    plot_domain = [-0.5, 0.5]
else:
    plot_domain = [-3.0, 0.25]


##############################
# Injection analysis to test
##############################

if morph == 'scatter':
    basedir = '/home/chad.henshaw/Injections/RIFT/dev/single_inject/hyperbolic/BBH-hyp_inject_mtot-40_q-2_E0-1.01_pphi0-4.4_dL-200_test_IGWN-POOL/'
    rundir = os.path.join(basedir, 'analysis_0/')
    
elif morph == 'spin-scatter':
    #basedir = '/home/chad.henshaw/Injections/RIFT/dev/single_inject/hyperbolic/scatter/BBH-hyp_inject_mtot-80_q-2_s1z-0.5_s2z-0.5_E0-1.02_pphi0-5.0_dL-250/'
    basedir = '/home/chad.henshaw/Injections/RIFT/dev/single_inject/hyperbolic/scatter/frame_test/posxi_scatter_updated-frames/'
    basedir2 = '/home/chad.henshaw/Injections/RIFT/dev/single_inject/hyperbolic/scatter/BBH-hyp_inject_mtot-80_q-2_s1z-0.5_s2z-0.5_E0-1.02_pphi0-5.0_dL-250/'
    rundir = os.path.join(basedir2, 'analysis_0/')
    
elif morph == 'capture':
    # mtot = 60 run. Note that the frames have been updated.
    #basedir = '/home/chad.henshaw/Injections/RIFT/dev/single_inject/hyperbolic/BBH-hyp_inject_mtot-60_E0-1.038_pphi0-4.7_dL-500_test_IGWN-POOL/'
    basedir = '/home/chad.henshaw/Injections/RIFT/dev/single_inject/hyperbolic/BBH-hyp_inject_mtot-60_E0-1.038_pphi0-4.7_dL-500_test_IGWN-POOL_updated-frames/'
    rundir = os.path.join(basedir, 'analysis_0/')
    
elif morph == 'capture2':
    # mtot = 40 run. Note that the frames have been updated.
    #basedir = '/home/chad.henshaw/Injections/RIFT/dev/single_inject/hyperbolic/BBH-hyp_inject_mtot-40_E0-1.038_pphi0-4.7_dL-500_test_IGWN-POOL/'
    basedir = '/home/chad.henshaw/Injections/RIFT/dev/single_inject/hyperbolic/BBH-hyp_inject_mtot-40_E0-1.038_pphi0-4.7_dL-500_test_IGWN-POOL_updated-frames/'
    rundir = os.path.join(basedir, 'analysis_0/')
    
elif morph == 'capturetest':
    basedir = '/home/chad.henshaw/RIFT_henshaw/dev/likelihood_compare/capture2_test/capture2_mtot-30_E0-1.038_pphi0-4.7_dL-500_test/'
    rundir = os.path.join(basedir, 'analysis_0/')
    
elif morph == 'capturetest2':
    basedir = '/home/chad.henshaw/RIFT_henshaw/dev/likelihood_compare/capture2_test/capture2_mtot-80_E0-1.038_pphi0-4.7_dL-500_test/'
    rundir = os.path.join(basedir, 'analysis_0/')
    
elif morph == 'posprec':
    basedir = '/home/chad.henshaw/Injections/RIFT/dev/single_inject/chiprms_tests/SEOBNRv4PHM/cust-chiprms-flat_puff_highSNR_prec_bbhinj-extr_q1-pos_OSG-ILE/'
    #basedir = '/home/chad.henshaw/education/injection_test/prod-def_highSNR_prec_bbhinj-extr_mtot-40_q1-pos_OSG-CIP/'
    rundir = os.path.join(basedir, 'analysis_0/')
    
elif morph == 'negprec':
    basedir = '/home/chad.henshaw/Injections/RIFT/dev/single_inject/chiprms_tests/SEOBNRv4PHM/cust-chiprms-flat_puff_highSNR_prec_bbhinj-extr_q1-neg_OSG-ILE/'
    rundir = os.path.join(basedir, 'analysis/')
    



# These are the evaluated grid points
#comp_file = os.path.join(rundir, 'all.net')

# Necessary paths for ILE calculation
cache_path = os.path.join(basedir, 'combined_frames/event_0/signals.cache')
PSD_path_H1 = os.path.join(rundir, 'H1-psd.xml.gz')
PSD_path_L1 = os.path.join(rundir, 'L1-psd.xml.gz')
dist_marg_path = os.path.join(basedir2, 'distance_marginalization_lookup.npz') # will probably need to put in a try/except for this

##############################
# Injected Signal
##############################

# These are the injected parameters
truth_xml = os.path.join(basedir, 'mdc.xml.gz')

# Load injected params into lalsimutils
P_truth = lalsimutils.xml_to_ChooseWaveformParams_array(truth_xml)[0]

# Build hoft from hlmoft
hlmT = {}
hlmF, hlmF_conj = lalsimutils.std_and_conj_hlmoff(P_truth, Lmax=lmax) # modes in fourier space
for mode in hlmF:
    hlmT[mode] = lalsimutils.DataInverseFourier(hlmF[mode]) # convert to time series
    
hlmoft_truth = lalsimutils.hoft_from_hlm(hlmT, P_truth, return_complex=True) # combine modes

hlmoft_truth_times = lalsimutils.evaluate_tvals(hlmoft_truth)



##############################
# User test waveform
##############################

if user_test:

    # Load injected params into lalsimutils
    P_test = P_truth.copy() # copy all settings from the truth file

    # override intrinsics with test values
    P_test.m1 = 53.34*SM
    P_test.m2 = 26.66*SM
    P_test.s1x = 0.0
    P_test.s1y = 0.0
    P_test.s1z = 0.60
    P_test.s2x = 0.0
    P_test.s2y = 0.0
    P_test.s2z = 0.60
    P_test.E0 = 1.02
    P_test.p_phi0 =  5.0

    P_test_list = [P_test]

    lalsimutils.ChooseWaveformParams_array_to_xml(P_test_list, fname='user_test')

    test_xml = os.path.join(os.getcwd(), 'user_test.xml.gz')

    # Build hoft from hlmoft
    hlmT_test = {}
    hlmF_test, hlmF_test_conj = lalsimutils.std_and_conj_hlmoff(P_test, Lmax=lmax) # modes in fourier space
    for mode in hlmF_test:
        hlmT_test[mode] = lalsimutils.DataInverseFourier(hlmF_test[mode]) # convert to time series

    hlmoft_test = lalsimutils.hoft_from_hlm(hlmT_test, P_test, return_complex=True) # combine modes

    hlmoft_test_times = lalsimutils.evaluate_tvals(hlmoft_test)

    print(f'Truth epoch is {hlmoft_truth.epoch}')
    print(f'Truth deltaT is {hlmoft_truth.deltaT}')
    print(f'Truth length is {hlmoft_truth.data.length}')

    print(f'Test epoch is {hlmoft_test.epoch}')
    print(f'Test deltaT is {hlmoft_test.deltaT}')
    print(f'Test length is {hlmoft_test.data.length}')


##############################
# Max Likelihood Waveform
##############################

if maxL:
    
    # These are the evaluated grid points
    comp_file = os.path.join(rundir, 'all.net')

    max_L = np.loadtxt(comp_file)[0] # load top row of all.net

    P_maxL = P_truth.copy() # copy all settings from the truth file

    # override intrinsics with max_L values

    P_maxL.m1 = max_L[1]*SM
    P_maxL.m2 = max_L[2]*SM
    P_maxL.s1x = max_L[3]
    P_maxL.s1y = max_L[4]
    P_maxL.s1z = max_L[5]
    P_maxL.s2x = max_L[6]
    P_maxL.s2y = max_L[7]
    P_maxL.s2z = max_L[8]
    P_maxL.E0 = max_L[9]
    P_maxL.p_phi0 = max_L[10]
    
    P_maxL_list = [P_maxL]
    
    lalsimutils.ChooseWaveformParams_array_to_xml(P_maxL_list, fname='maxL')

    maxL_xml = os.path.join(os.getcwd(), 'maxL.xml.gz')

    # Plot h(t) from lalsimutils.hoft - max L waveform
    #hpoft_maxL = lalsimutils.hoft(P_maxL, Lmax=lmax, Fp=1, Fc=0)
    #hcoft_maxL = lalsimutils.hoft(P_maxL, Lmax=lmax, Fp=0, Fc=1)

    #hoft_maxL = hpoft_maxL.data.data + 1j*(-1)*hcoft_maxL.data.data
    #hoft_maxL_times = lalsimutils.evaluate_tvals(hpoft_maxL)

    # Build hoft from hlmoft
    hlmT_maxL = {}
    hlmF_maxL, hlmF_maxL_conj = lalsimutils.std_and_conj_hlmoff(P_maxL, Lmax=lmax) # modes in fourier space
    for mode in hlmF_maxL:
        hlmT_maxL[mode] = lalsimutils.DataInverseFourier(hlmF_maxL[mode]) # convert to time series

    hlmoft_maxL = lalsimutils.hoft_from_hlm(hlmT_maxL, P_maxL, return_complex=True) # combine modes

    hlmoft_maxL_times = lalsimutils.evaluate_tvals(hlmoft_maxL)




##############################
# ILE calculation
##############################

new_data_start = 999999994.0
new_data_end = 1000000002.0
new_seglen = int(new_data_end - new_data_start)

# sim_xml is the path to the xml with the point you want to try
def run_ILE(sim_xml):

    if sim_xml == truth_xml:
        print('Evaluating true parameters')
    elif sim_xml == test_xml:
        print('Evaluating test parameters')
    elif sim_xml == maxL_xml:
        print('Evaluating maxL parameters')

    # extract ILE settings from command-single
    cs_path = os.path.join(rundir, 'command-single.sh')
    
    with open(cs_path, 'r') as file:
        lines = file.readlines()
    
    # Skip the first two lines and get the options line
    options_line = lines[2].strip()
    
    # Dictionary of new values for specific options
    new_values = {
        '--cache': cache_path,
        '--psd-file H1=': PSD_path_H1,
        '--psd-file L1=': PSD_path_L1,
        '--distance-marginalization-lookup-table': dist_marg_path,
        '--sim-xml': sim_xml,
        '--n-events-to-analyze': '1',
        '--l-max': lmax,
        '--data-start-time': new_data_start,
        '--data-end-time': new_data_end
    }
    
    
    # Split the line into parts, keeping paired options together
    parts = options_line.split()
    options = []
    i = 1  # Start from the first element after the executable path
    
    while i < len(parts):
        if parts[i].startswith('--'):
            if i + 1 < len(parts) and not parts[i + 1].startswith('--'):
                option = parts[i]
                value = parts[i + 1]
                if option == '--psd-file':
                    if 'H1=' in value:
                        value = f'H1={new_values["--psd-file H1="]}'
                    elif 'L1=' in value:
                        value = f'L1={new_values["--psd-file L1="]}'
                elif option in new_values:
                    value = new_values[option]
                options.append(f'{option} {value}')
                i += 2
            else:
                option = parts[i]
                if option in new_values:
                    options.append(f'{option} {new_values[option]}')
                else:
                    options.append(option)
                i += 1
        else:
            i += 1
    
    # Add new options
    options.append('--force-xpy')
    if sim_xml == truth_xml:
        options.append('--output-file truth_output.xml')
    elif sim_xml == test_xml:
        options.append('--output-file test_output.xml')
    elif sim_xml == maxL_xml:
        options.append('--output-file maxL_output.xml')
    #options.append('--save-samples')
    #options.append('--internal-use-lnL')
    #options.append('--sampler-method GMM')
    
    # remove options
    #options.remove('--internal-waveform-fd-L-frame')
    
    # ILE exe path:
    ILE_path = os.path.join(os.path.dirname(sys.executable), 'integrate_likelihood_extrinsic_batchmode')
    
    # Generate the command string
    cmd = 'python ' + ILE_path + ' ' + ' '.join(options)
    print(cmd)
    
    os.system(cmd)

    if sim_xml == truth_xml:
        output_file_path = 'truth_output.xml_0_.dat'
    elif sim_xml == test_xml:
        output_file_path = 'test_output.xml_0_.dat'
        
    elif sim_xml == maxL_xml:
        output_file_path = 'maxL_output.xml_0_.dat'

    # print L values
    L_value = np.loadtxt(output_file_path)[11]

    print(f'The L value is {L_value}')

    

    return L_value

if calc_ILE:

    truth_L = run_ILE(truth_xml)
    
    if user_test:
        test_L = run_ILE(test_xml)
    
    if maxL:
        maxL_L = run_ILE(maxL_xml)

    print(f'The L value for the true params is {truth_L}')
    print(f'The L value for the test params is {test_L}')
    if maxL and user_test:
        print(f'The L value for the maxL params is {maxL_L}')
        result_list = [truth_L, test_L, maxL_L]
    elif maxL:
        result_list = [truth_L, maxL_L]
    else:
        result_list = [truth_L]

    if maxL and user_test:
        result_file = f'{morph}_lmax-{lmax}_Injection+maxL+test_mtot-{(P_test.m1+P_test.m2)/SM}_q-{P_test.m1/P_test.m2}_chi1-{P_test.s1z}_chi2-{P_test.s2z}_E0-{P_test.E0}_pphi0-{P_test.p_phi0}_ILEresults.txt'
    elif maxL:
        result_file = f'{morph}_lmax-{lmax}_Injection+maxL_ILEresults.txt'
    else:
        result_file = f'{morph}_lmax-{lmax}_Injection_ILEresults.txt'

    # Open the file in write mode
    with open(result_file, "w") as file:
        # Write each item in the list to the file
        for item in result_list:
            file.write(f"{item}\n")

# Note that this is the output format:
#event_id, m1, m2, P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z,  P.E0, P.p_phi0, log_res+manual_avoid_overflow_logarithm, sqrt_var_over_res,sampler.ntotal, neff
# the number that we're interested in is log_res+manual_avoid_overflow_logarithm, so column 11

#output_file_path = 'test_output.xml_0_.dat'

else:
    # try to load the results from the text file
    if maxL and user_test:
        result_file = f'{morph}_lmax-{lmax}_Injection+maxL+test_mtot-{(P_test.m1+P_test.m2)/SM}_q-{P_test.m1/P_test.m2}_chi1-{P_test.s1z}_chi2-{P_test.s2z}_E0-{P_test.E0}_pphi0-{P_test.p_phi0}_ILEresults.txt'
    elif maxL:
        result_file = f'{morph}_lmax-{lmax}_Injection+maxL_ILEresults.txt'
    else:
        result_file = f'{morph}_lmax-{lmax}_Injection_ILEresults.txt'

    try:
        with open(result_file, "r") as file:
            # Read each line in the file and convert it to a float
            result_list = [float(line.strip()) for line in file]
            truth_L = result_list[0]
            test_L = result_list[1]
            maxL_L = result_list[2]
    except FileNotFoundError:
        print('No result file found!')
        
        


##############################
# Plotting
##############################

fig, axs = plt.subplots(figsize=(20, 10), nrows=2, ncols=1)
ax = axs.flatten()

# Injected signal and test signal from recombined hlmoft
ax[0].plot(hlmoft_truth_times, np.real(hlmoft_truth.data.data), color='blue', alpha=0.3, label='Injected Signal: hlmoft')

if user_test:
    ax[0].plot(hlmoft_test_times, np.real(hlmoft_test.data.data), color='orange', linestyle='dotted', label='Test waveform: hlmoft')

if maxL:
    ax[0].plot(hlmoft_maxL_times, np.real(hlmoft_maxL.data.data), color='red', linestyle='dashed', label='MaxL waveform: hlmoft')


ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Strain')

# Enable the legend
ax[0].legend()
ax[0].grid(True, alpha=0.4)

ax[0].set_xlim(plot_domain)

textstr_inj = '\n'.join((
    'Injected:',
    '$m_1 = %.1f \: [M_{\odot}]$' % (P_truth.m1/SM),
    '$m_2 = %.1f \: [M_{\odot}]$' % (P_truth.m2/SM),
    '$S_{1,z}=%.2f$' % (P_truth.s1z),
    '$S_{2,z}=%.2f$' % (P_truth.s2z),
    '$E_0=%.5f$' % (P_truth.E0),
    '$p_{\phi}^0=%.5f$' % (P_truth.p_phi0)))

ax[0].text(0.0200, 0.98, textstr_inj, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

if user_test:
    textstr_test = '\n'.join((
        'Test waveform:',
        '$m_1 = %.1f \: [M_{\odot}]$' % (P_test.m1/SM),
        '$m_2 = %.1f \: [M_{\odot}]$' % (P_test.m2/SM),
        '$S_{1,z}=%.2f$' % (P_test.s1z),
        '$S_{2,z}=%.2f$' % (P_test.s2z),
        '$E_0=%.5f$' % (P_test.E0),
        '$p_{\phi}^0=%.5f$' % (P_test.p_phi0)))

    ax[0].text(0.2200, 0.98, textstr_test, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

if maxL:
    textstr_maxL = '\n'.join((
        'Max Likelihood:',
        '$m_1 = %.1f \: [M_{\odot}]$' % (P_maxL.m1/SM),
        '$m_2 = %.1f \: [M_{\odot}]$' % (P_maxL.m2/SM),
        '$S_{1,z}=%.2f$' % (P_maxL.s1z),
        '$S_{2,z}=%.2f$' % (P_maxL.s2z),
        '$E_0=%.5f$' % (P_maxL.E0),
        '$p_{\phi}^0=%.5f$' % (P_maxL.p_phi0)))

    ax[0].text(0.1200, 0.98, textstr_maxL, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

lmax_str = f'lmax = {lmax}' 

ax[0].text(0.01, 0.01, lmax_str, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# Injected signal and test signal from recombined hlmoft
ax[1].plot(hlmoft_truth_times, np.real(hlmoft_truth.data.data), color='blue', alpha=0.5, label='Injected Signal: hlmoft')

if user_test:
    ax[1].plot(hlmoft_test_times, np.real(hlmoft_test.data.data), color='orange', label='Test waveform: hlmoft')

if maxL:
    ax[1].plot(hlmoft_maxL_times, np.real(hlmoft_maxL.data.data), color='red', linestyle='dashed', label='MaxL waveform: hlmoft')


ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Strain')

# Enable the legend
ax[1].legend()
ax[1].grid(True, alpha=0.4)

try:
    if maxL and user_test:
        L_str = '\n'.join((
    f'Truth L value: {truth_L}',
    f'Test L value: {test_L}',
    f'maxL L value: {maxL_L}'))
    elif maxL:
        L_str = '\n'.join((
    f'Truth L value: {truth_L}',
    f'maxL L value: {maxL_L}'))
    else:
        L_str = f'Truth L value: {truth_L}'
        
    ax[1].text(0.01, 0.01, L_str, transform=ax[1].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
except:
    print('something wrong')
    
    

    
if maxL and user_test:
    plt.savefig(f'./spin/{morph}_lmax-{lmax}_Injection+maxL+test_mtot-{(P_test.m1+P_test.m2)/SM}_q-{P_test.m1/P_test.m2}_chi1-{P_test.s1z}_chi2-{P_test.s2z}_E0-{P_test.E0}_pphi0-{P_test.p_phi0}_seglen-{new_seglen}s_post-tapering-fix.png', bbox_inches='tight')
elif maxL:
    plt.savefig(f'{morph}_lmax-{lmax}_Injection+maxL.png', bbox_inches='tight')
else:
    plt.savefig(f'{morph}_lmax-{lmax}_Injection.png', bbox_inches='tight')


































#if os.path.isfile(cache_path):
#    print("File exists")
#else:
#    print("File does not exist")

#ILE settings (we could probably pull these from command-single itself)
#cache file
#PSDs and channel names
#distance marg lookup table
#the mdc xml






# Load injected params into lalsimutils
#P_truth = lalsimutils.xml_to_ChooseWaveformParams_array(truth_xml)[0]

    

    

##############################
# Max Likelihood Waveform
##############################

#if maxL:

#    max_L = np.loadtxt(comp_file)[0] # load top row of all.net

#    P_maxL = P_truth.copy() # copy all settings from the truth file

    # override intrinsics with max_L values

#    P_maxL.m1 = max_L[1]*SM
#    P_maxL.m2 = max_L[2]*SM
#    P_maxL.s1x = max_L[3]
#    P_maxL.s1y = max_L[4]
#    P_maxL.s1z = max_L[5]
#    P_maxL.s2x = max_L[6]
#    P_maxL.s2y = max_L[7]
#    P_maxL.s2z = max_L[8]
#    P_maxL.E0 = max_L[9]
#    P_maxL.p_phi0 = max_L[10]

    # Plot h(t) from lalsimutils.hoft - max L waveform
#    hpoft_maxL = lalsimutils.hoft(P_maxL, Lmax=lmax, Fp=1, Fc=0)
#    hcoft_maxL = lalsimutils.hoft(P_maxL, Lmax=lmax, Fp=0, Fc=1)

#    hoft_maxL = hpoft_maxL.data.data + 1j*(-1)*hcoft_maxL.data.data
#    hoft_maxL_times = lalsimutils.evaluate_tvals(hpoft_maxL)

    # Build hoft from hlmoft
#    hlmT_maxL = {}
#    hlmF_maxL, hlmF_maxL_conj = lalsimutils.std_and_conj_hlmoff(P_maxL, Lmax=lmax) # modes in fourier space
#    for mode in hlmF_maxL:
#        hlmT_maxL[mode] = lalsimutils.DataInverseFourier(hlmF_maxL[mode]) # convert to time series

#    hlmoft_maxL = lalsimutils.hoft_from_hlm(hlmT_maxL, P_maxL, return_complex=True) # combine modes

#    hlmoft_maxL_times = lalsimutils.evaluate_tvals(hlmoft_maxL)



#if test:

#    P_test = P_truth.copy() # copy all settings from the truth file

    # override intrinsics with max_L values

#    P_test.m1 = 21.0*SM
#    P_test.m2 = 20.0*SM
#    P_test.s1x = 0.0
#    P_test.s1y = 0.0
#    P_test.s1z = 0.0
#    P_test.s2x = 0.0
#    P_test.s2y = 0.0
#    P_test.s2z = 0.0
#    P_test.E0 = 1.0099999904632568
#    P_test.p_phi0 =  4.400000095367432

    # Plot h(t) from lalsimutils.hoft - test waveform
#    hoft_test = lalsimutils.hoft(P_test, Lmax=lmax)
#    hoft_test_times = lalsimutils.evaluate_tvals(hoft_test)

    # Build hoft from hlmoft
#    hlmT_test = {}
#    hlmF_test, hlmF_test_conj = lalsimutils.std_and_conj_hlmoff(P_test, Lmax=lmax) # modes in fourier space
#    for mode in hlmF_test:
#        hlmT_test[mode] = lalsimutils.DataInverseFourier(hlmF_test[mode]) # convert to time series

#    hlmoft_test = lalsimutils.hoft_from_hlm(hlmT_test, P_test) # combine modes

#    hlmoft_test_times = lalsimutils.evaluate_tvals(hlmoft_test)



