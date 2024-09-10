#!/usr/bin/env python3

# Comparing likelihood values for an injection

#import RIFT.lalsimutils_test as lalsimutils
import RIFT.lalsimutils as lalsimutils

import os, sys, ast

import numpy as np
from matplotlib import pyplot as plt
import lal
import argparse
import re
import configparser
config = configparser.ConfigParser(allow_no_value=True)
natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

SM = lal.MSUN_SI
lsu_PC=lal.PC_SI

##############################
# Options and Settings
##############################

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for comparing likelihood values for an injection.')
    parser.add_argument('--lmax', type=int, default=4, help='Maximum value of L.')
    parser.add_argument('--calc-ILE', action='store_true', help='Flag to calculate ILE.')
    parser.add_argument('--user-test', action='store_true', help='Flag to run user test.')
    parser.add_argument('--maxL', action='store_true', help='Flag to calculate maxL.')
    parser.add_argument('--analysis-dir', required=True, type=str, help='Absolute path to the analysis directory.')
    parser.add_argument('--plot-domain', default="[-6.0, 2.0]", type=str, help='Plot time domain')
    parser.add_argument('--use-hyperbolic', action='store_true', help='Configure for hyperbolic analysis')
    parser.add_argument('--hlmoff', action='store_true', help='Builds h(t) from hlm(f) instead of hlm(t)')
    parser.add_argument('--fast-ILE', action='store_true', help='Runs ILE with low n-max, for speed')
    parser.add_argument('--test-from-log', action='store_true', help='Loads a random draw from exclusions_log_test.txt')
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

plot_domain = ast.literal_eval(args.plot_domain)
if args.use_hyperbolic:
    plot_domain = [-0.5, 0.5]

if args.user_test and args.test_from_log:
    print('CANNOT use both user-test and test-from-log, setting test-from-log to false')
    args.test_from_log = False

rundir = args.analysis_dir
basedir = os.path.dirname(rundir)

# Waveform options not contained in xml
config_file = os.path.join(rundir, 'local.ini')
if not(os.path.exists(config_file)):
    print(f"Config file not found at: {config_file}")
    sys.exit(1)

config.read(config_file)
srate = float(config.get('engine', 'srate'))

# Necessary paths for ILE calculation
cache_path = os.path.join(basedir, 'combined_frames/event_0/signals.cache')
PSD_path_H1 = os.path.join(rundir, 'H1-psd.xml.gz')
PSD_path_L1 = os.path.join(rundir, 'L1-psd.xml.gz')
dist_marg_path = os.path.join(basedir, 'distance_marginalization_lookup.npz') # will probably need to put in a try/except for this

# Create diagnostic directory
diag_path = os.path.join(rundir, 'diagnostic')
if not(os.path.exists(diag_path)):
    os.mkdir(diag_path)
    
## Determine deltaF from PSDs
# Note that for the PSDs we're currently using, deltaF = 0.0625.
# just going to read from H1, assuming that L1 is the same
# (this is true for the O4 design PSDs)
deltaF = lalsimutils.get_psd_series_from_xmldoc(PSD_path_H1, 'H1').deltaF


##############################
# Function definitions
##############################

def gen_wav(xml):
    # load xml
    P = lalsimutils.xml_to_ChooseWaveformParams_array(xml)[0]
    # update options
    P.deltaT = 1./srate
    P.deltaF = deltaF
    P.print_params()
    if not(args.use_hyperbolic):
        P.taper=lalsimutils.lsu_TAPER_START
    P.print_params()
    # build hoft from hlmoft    
    if args.hlmoff:
        hlmT = {}
        hlmF, hlmF_conj = lalsimutils.std_and_conj_hlmoff(P, Lmax=args.lmax) # modes in fourier space
        for mode in hlmF:
            hlmT[mode] = lalsimutils.DataInverseFourier(hlmF[mode]) # convert to time series
    else:
        hlmT = lalsimutils.hlmoft(P, Lmax=args.lmax)
        #hlmT = lalsimutils.hlmoft(P, Lmax=args.lmax, no_condition=True)
        
    hoft = lalsimutils.hoft_from_hlm(hlmT, P, return_complex=True) # combine modes
    hoft_times = lalsimutils.evaluate_tvals(hoft)
    
    return hoft, hoft_times
    
def run_ILE(xml, output_name):
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
        '--sim-xml': xml,
        '--n-events-to-analyze': '1',
        '--l-max': args.lmax
    }
    
    if args.fast_ILE:
        new_values['--n-max'] = 40000
    
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
    output_path = os.path.join(diag_path, f'{output_name}.xml')
    options.append(f'--output-file {output_path}')
    #options.append('--save-samples')
    #options.append('--internal-use-lnL')
    #options.append('--sampler-method GMM')
    
    # remove options
    #options.remove('--internal-waveform-fd-L-frame')
    
    # ILE exe path:
    ILE_path = os.path.join(os.path.dirname(sys.executable), 'integrate_likelihood_extrinsic_batchmode')
    #ILE_path = '/home/chad.henshaw/virtualenvs/prectest_rift_O4c/bin/integrate_likelihood_extrinsic_batchmode_test'
    
    # Generate the command string
    cmd = 'python ' + ILE_path + ' ' + ' '.join(options)
    print(cmd)    
    os.system(cmd)

    # print L values
    if args.use_hyperbolic:
        lnL_value = np.loadtxt(output_path + '_0_.dat')[11]
    else:
        lnL_value = np.loadtxt(output_path + '_0_.dat')[9]

    print(f'The lnL value is {lnL_value}')    

    return lnL_value
    
    

##############################
# Injected Signal
##############################

print('Now generating true param waveform')

# These are the injected parameters
truth_xml = os.path.join(basedir, 'mdc.xml.gz')

# Generate waveform
hoft_truth, hoft_truth_times = gen_wav(truth_xml)

# Save ChooseWaveformParams object for later use
P_truth = lalsimutils.xml_to_ChooseWaveformParams_array(truth_xml)[0]
P_truth.deltaT = 1./srate


##############################
# User test waveform
##############################

if args.user_test:
    
    print('Now generating test param waveform')

    # Load injected params into lalsimutils
    P_test = P_truth.copy() # copy all settings from the truth file

    # override intrinsics with test values
    P_test.m1 = 26.66*SM
    P_test.m2 = 13.34*SM
    P_test.s1x = 0.0
    P_test.s1y = 0.0
    P_test.s1z = 0.60
    P_test.s2x = 0.0
    P_test.s2y = 0.0
    P_test.s2z = 0.60
    if args.use_hyperbolic:
        P_test.E0 = 1.02
        P_test.p_phi0 =  5.0

    P_test_list = [P_test]

    lalsimutils.ChooseWaveformParams_array_to_xml(P_test_list, fname=os.path.join(diag_path,'user_test'))

    test_xml = os.path.join(diag_path, 'user_test.xml.gz')
    
    hoft_test, hoft_test_times = gen_wav(test_xml)
    
if args.test_from_log:
    # try to load log file
    log_path = os.path.join(args.analysis_dir, 'diagnostic/exclusions_log_test.txt')
    
    try:
        if not os.path.isfile(log_path):
            raise FileNotFoundError(f"No exlcusions log found! Skipping!")
    
        data = np.loadtxt(log_path)
        rand_line = data[np.random.choice(data.shape[0])]

        # Load injected params into lalsimutils
        P_test = P_truth.copy() # copy all settings from the truth file

        # assign values from log to lalsimutils
        P_test.m1 = float(rand_line[1])*SM
        P_test.m2 = float(rand_line[2])*SM
        P_test.s1x = float(rand_line[3])
        P_test.s1y = float(rand_line[4])
        P_test.s1z = float(rand_line[5])
        P_test.s2x = float(rand_line[6])
        P_test.s2y = float(rand_line[7])
        P_test.s2z = float(rand_line[8])
        if args.use_hyperbolic:
            P_test.E0 = float(rand_line[9]) # NEEDS TO BE TESTED!!
            P_test.p_phi0 = float(rand_line[10]) # NEEDS TO BE TESTED!!
            
        P_test_list = [P_test]

        lalsimutils.ChooseWaveformParams_array_to_xml(P_test_list, fname=os.path.join(diag_path,'user_test'))

        test_xml = os.path.join(diag_path, 'user_test.xml.gz')

        hoft_test, hoft_test_times = gen_wav(test_xml)
    
    
    except FileNotFoundError as e:
        print(e)
        pass



##############################
# Max Likelihood Waveform
##############################

if args.maxL:
    
    print('Now generating maxL param waveform')
    
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
    if args.use_hyperbolic:
        P_maxL.E0 = max_L[9]
        P_maxL.p_phi0 = max_L[10]
    
    P_maxL_list = [P_maxL]
    
    lalsimutils.ChooseWaveformParams_array_to_xml(P_maxL_list, fname=os.path.join(diag_path,'maxL'))

    maxL_xml = os.path.join(diag_path, 'maxL.xml.gz')

    hoft_maxL, hoft_maxL_times = gen_wav(maxL_xml)




##############################
# ILE calculation
##############################

#new_data_start = 999999994.0
#new_data_end = 1000000002.0
#new_seglen = int(new_data_end - new_data_start)

if args.calc_ILE:

    truth_lnL = run_ILE(truth_xml, 'truth_output')
    print(f'The lnL value for the true params is {truth_lnL}')
    
    if args.user_test:
        test_lnL = run_ILE(test_xml, 'test_output')
        print(f'The lnL value for the test params is {test_lnL}')
    
    if args.maxL:
        maxL_lnL = run_ILE(maxL_xml, 'maxL_output')
        print(f'The lnL value for the maxL params is {maxL_lnL}')

    
    if args.maxL and args.user_test:        
        result_list = [truth_lnL, test_lnL, maxL_lnL]
        if args.use_hyperbolic:
            result_file = os.path.join(diag_path, f'Injection+maxL+test_mtot-{(P_test.m1+P_test.m2)/SM}_q-{P_test.m1/P_test.m2}_chi1-{P_test.s1z}_chi2-{P_test.s2z}_E0-{P_test.E0}_pphi0-{P_test.p_phi0}_lmax-{args.lmax}_ILEresults.txt')
        else:
            result_file = os.path.join(diag_path, f'Injection+maxL+test_mtot-{(P_test.m1+P_test.m2)/SM}_q-{P_test.m1/P_test.m2}_chi1-{P_test.s1z}_chi2-{P_test.s2z}_lmax-{args.lmax}_ILEresults.txt')
                                       
    elif args.maxL and not(args.user_test):
        result_list = [truth_lnL, maxL_lnL]
        result_file = os.path.join(diag_path, f'Injection+maxL_lmax-{args.lmax}_ILEresults.txt')
        
    elif args.user_test and not(args.maxL):
        result_list = [truth_lnL, test_lnL]
        if args.use_hyperbolic:
            result_file = os.path.join(diag_path, f'Injection+test_mtot-{(P_test.m1+P_test.m2)/SM}_q-{P_test.m1/P_test.m2}_chi1-{P_test.s1z}_chi2-{P_test.s2z}_E0-{P_test.E0}_pphi0-{P_test.p_phi0}_lmax-{args.lmax}_ILEresults.txt')
        else:
            result_file = os.path.join(diag_path, f'Injection+test_mtot-{(P_test.m1+P_test.m2)/SM}_q-{P_test.m1/P_test.m2}_chi1-{P_test.s1z}_chi2-{P_test.s2z}_lmax-{args.lmax}_ILEresults.txt')
    else:
        result_list = [truth_lnL]
        result_file = os.path.join(diag_path, f'Injection_lmax-{args.lmax}_ILEresults.txt')

    # Open the file in write mode
    with open(result_file, "w") as file:
        # Write each item in the list to the file
        for item in result_list:
            file.write(f"{item}\n")


##############################
# Plotting
##############################

fig, axs = plt.subplots(figsize=(20, 10), nrows=2, ncols=1)
ax = axs.flatten()

# Injected signal and test signal from recombined hlmoft
ax[0].plot(hoft_truth_times, np.real(hoft_truth.data.data), color='blue', alpha=0.3, label=r'Injected Signal: $h_+(t)$')

if args.user_test:
    ax[0].plot(hoft_test_times, np.real(hoft_test.data.data), color='green', linestyle='dotted', label=r'Test waveform: $h_+(t)$')

if args.maxL:
    ax[0].plot(hoft_maxL_times, np.real(hoft_maxL.data.data), color='red', linestyle='dashed', label=r'Max lnL waveform: $h_+(t)$')


ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Strain')

# Enable the legend
ax[0].legend()
ax[0].grid(True, alpha=0.4)

ax[0].set_xlim(plot_domain)

if args.use_hyperbolic:

    textstr_inj = '\n'.join((
        'Injected:',
        '$m_1 = %.1f \: [M_{\odot}]$' % (P_truth.m1/SM),
        '$m_2 = %.1f \: [M_{\odot}]$' % (P_truth.m2/SM),
        '$S_{1,z}=%.2f$' % (P_truth.s1z),
        '$S_{2,z}=%.2f$' % (P_truth.s2z),
        '$E_0=%.5f$' % (P_truth.E0),
        '$p_{\phi}^0=%.5f$' % (P_truth.p_phi0)))

    ax[0].text(0.0200, 0.98, textstr_inj, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if args.user_test:
        textstr_test = '\n'.join((
            'Test waveform:',
            '$m_1 = %.1f \: [M_{\odot}]$' % (P_test.m1/SM),
            '$m_2 = %.1f \: [M_{\odot}]$' % (P_test.m2/SM),
            '$S_{1,z}=%.2f$' % (P_test.s1z),
            '$S_{2,z}=%.2f$' % (P_test.s2z),
            '$E_0=%.5f$' % (P_test.E0),
            '$p_{\phi}^0=%.5f$' % (P_test.p_phi0)))

        ax[0].text(0.2200, 0.98, textstr_test, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if args.maxL:
        textstr_maxL = '\n'.join((
            'Max Likelihood:',
            '$m_1 = %.1f \: [M_{\odot}]$' % (P_maxL.m1/SM),
            '$m_2 = %.1f \: [M_{\odot}]$' % (P_maxL.m2/SM),
            '$S_{1,z}=%.2f$' % (P_maxL.s1z),
            '$S_{2,z}=%.2f$' % (P_maxL.s2z),
            '$E_0=%.5f$' % (P_maxL.E0),
            '$p_{\phi}^0=%.5f$' % (P_maxL.p_phi0)))

        ax[0].text(0.1200, 0.98, textstr_maxL, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
else:
    textstr_inj = '\n'.join((
        'Injected:',
        '$m_1 = %.1f \: [M_{\odot}]$' % (P_truth.m1/SM),
        '$m_2 = %.1f \: [M_{\odot}]$' % (P_truth.m2/SM),
        '$S_{1,x}=%.2f$' % (P_truth.s1x),
        '$S_{1,y}=%.2f$' % (P_truth.s1y),
        '$S_{1,z}=%.2f$' % (P_truth.s1z),
        '$S_{2,x}=%.2f$' % (P_truth.s2x),
        '$S_{2,y}=%.2f$' % (P_truth.s2y),
        '$S_{2,z}=%.2f$' % (P_truth.s2z),
        '$\chi_p=%.2f$' % (P_truth.extract_param('chi_p')),
        r'$\sqrt{\langle \chi_p^2 \rangle} =%.2f$' % (P_truth.extract_param('chi_prms'))))

    ax[0].text(0.0200, 0.98, textstr_inj, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if args.user_test:
        textstr_test = '\n'.join((
            'Test waveform:',
            '$m_1 = %.1f \: [M_{\odot}]$' % (P_test.m1/SM),
            '$m_2 = %.1f \: [M_{\odot}]$' % (P_test.m2/SM),
            '$S_{1,x}=%.2f$' % (P_test.s1x),
            '$S_{1,y}=%.2f$' % (P_test.s1y),
            '$S_{1,z}=%.2f$' % (P_test.s1z),
            '$S_{2,x}=%.2f$' % (P_test.s2x),
            '$S_{2,y}=%.2f$' % (P_test.s2y),
            '$S_{2,z}=%.2f$' % (P_test.s2z),
            '$\chi_p=%.2f$' % (P_test.extract_param('chi_p')),
            r'$\sqrt{\langle \chi_p^2 \rangle} =%.2f$' % (P_truth.extract_param('chi_prms'))))

        ax[0].text(0.2200, 0.98, textstr_test, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if args.maxL:
        textstr_maxL = '\n'.join((
            'Max Likelihood:',
            '$m_1 = %.1f \: [M_{\odot}]$' % (P_maxL.m1/SM),
            '$m_2 = %.1f \: [M_{\odot}]$' % (P_maxL.m2/SM),
            '$S_{1,x}=%.2f$' % (P_maxL.s1x),
            '$S_{1,y}=%.2f$' % (P_maxL.s1y),
            '$S_{1,z}=%.2f$' % (P_maxL.s1z),
            '$S_{2,x}=%.2f$' % (P_maxL.s2x),
            '$S_{2,y}=%.2f$' % (P_maxL.s2y),
            '$S_{2,z}=%.2f$' % (P_maxL.s2z),
            '$\chi_p=%.2f$' % (P_maxL.extract_param('chi_p')),
            r'$\sqrt{\langle \chi_p^2 \rangle }=%.2f$' % (P_maxL.extract_param('chi_prms'))))

        ax[0].text(0.1200, 0.98, textstr_maxL, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

lmax_str = f'lmax = {args.lmax}' 

ax[0].text(0.01, 0.01, lmax_str, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# Injected signal and test signal from recombined hlmoft
ax[1].plot(hoft_truth_times, np.real(hoft_truth.data.data), color='blue', alpha=0.5, label=r'Injected Signal: $h_+(t)$')

if args.user_test:
    ax[1].plot(hoft_test_times, np.real(hoft_test.data.data), color='green', label=r'Test waveform: $h_+(t)$')

if args.maxL:
    ax[1].plot(hoft_maxL_times, np.real(hoft_maxL.data.data), color='red', linestyle='dashed', label=r'MaxL waveform: $h_+(t)$')


ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Strain')

# Enable the legend
ax[1].legend()
ax[1].grid(True, alpha=0.4)

try:
    if args.maxL and args.user_test:
        L_str = '\n'.join((
    f'Truth lnL value: {truth_lnL}',
    f'Test lnL value: {test_lnL}',
    f'maxL lnL value: {maxL_lnL}'))
    elif args.maxL and not(args.user_test):
        L_str = '\n'.join((
    f'Truth lnL value: {truth_lnL}',
    f'maxL lnL value: {maxL_lnL}'))
    elif args.user_test and not(args.maxL):
        L_str = '\n'.join((
    f'Truth lnL value: {truth_lnL}',
    f'Test lnL value: {test_lnL}'))
    else:
        L_str = f'Truth lnL value: {truth_lnL}'
        
    ax[1].text(0.01, 0.01, L_str, transform=ax[1].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
except:
    print('something wrong')


if args.maxL and args.user_test:
    if args.use_hyperbolic:
        savepath = os.path.join(diag_path, f'Injection+maxL+test_mtot-{(P_test.m1+P_test.m2)/SM}_q-{P_test.m1/P_test.m2}_chi1-{P_test.s1z}_chi2-{P_test.s2z}_E0-{P_test.E0}_pphi0-{P_test.p_phi0}_lmax-{args.lmax}_ILEresults.png')
    else:
        savepath = os.path.join(diag_path, f'Injection+maxL+test_mtot-{(P_test.m1+P_test.m2)/SM}_q-{P_test.m1/P_test.m2}_chi1-{P_test.s1z}_chi2-{P_test.s2z}_lmax-{args.lmax}_ILEresults.png')
                                       
elif args.maxL and not(args.user_test):
    savepath = os.path.join(diag_path, f'Injection+maxL_lmax-{args.lmax}_ILEresults.png')

elif args.user_test and not(args.maxL):
    if args.use_hyperbolic:
        savepath = os.path.join(diag_path, f'Injection+test_mtot-{(P_test.m1+P_test.m2)/SM}_q-{P_test.m1/P_test.m2}_chi1-{P_test.s1z}_chi2-{P_test.s2z}_E0-{P_test.E0}_pphi0-{P_test.p_phi0}_lmax-{args.lmax}_ILEresults.png')
    else:
        savepath = os.path.join(diag_path, f'Injection+test_mtot-{(P_test.m1+P_test.m2)/SM}_q-{P_test.m1/P_test.m2}_chi1-{P_test.s1z}_chi2-{P_test.s2z}_lmax-{args.lmax}_ILEresults.png')
else:
    savepath = os.path.join(diag_path, f'Injection_lmax-{args.lmax}_ILEresults.png')
        
        
plt.savefig(savepath, bbox_inches='tight')