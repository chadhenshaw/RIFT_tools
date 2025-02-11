#!/usr/bin/env python3

# Comparing likelihood values for an injection

#import RIFT.lalsimutils_test as lalsimutils
import RIFT.lalsimutils as lalsimutils
#import RIFT.likelihood.factored_likelihood as factored_likelihood  # direct hoft call
import os, sys, ast
import shutil

import numpy as np
from matplotlib import pyplot as plt
import lal
import argparse
import re
import configparser
import lalframe
from gwpy.timeseries import TimeSeries
config = configparser.ConfigParser(allow_no_value=True)
natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

SM = lal.MSUN_SI
lsu_PC=lal.PC_SI
lsu_DimensionlessUnit = lal.DimensionlessUnit

##############################
# Options and Settings
##############################

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for comparing likelihood values for an injection.')
    parser.add_argument('--analysis-dir', required=True, type=str, help='Absolute path to the analysis directory.')
    parser.add_argument('--calc-ILE', action='store_true', help='Flag to calculate ILE.')
    parser.add_argument('--user-test', action='store_true', help='Flag to run user test.')
    parser.add_argument('--maxL', action='store_true', help='Flag to calculate maxL.')
    parser.add_argument('--fast-ILE', action='store_true', help='Runs ILE with low n-max, for speed')
    parser.add_argument("--no-plots", action='store_true', help='Does not generate or plot waveforms - for use with --calc-ILE')
    parser.add_argument('--plot-topN', type=int, default=0, help='Plot waveforms for the top N rows (excluding maxL) from all.net. Requires --maxL.')
    parser.add_argument('--pin-extrinsics', action='store_true', help='Pins RA, DEC, tref to their true values for ILE calculation')
    parser.add_argument('--verbose-ILE',action='store_true',help='Adds --verbose to ILE')
    parser.add_argument('--pin-distance', action='store_true', help='Pins the luminosity distance to its true value ILE calculation')
    #parser.add_argument('--make-L1-frame', action='store_true', help='Makes a new frame with L1 ideal sky location')
    #parser.add_argument('--make-22-frame', action='store_true', help='Makes new frames with just 22 modes')
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

plot_domain = [-0.5, 0.5]

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

# Necessary paths for ILE calculation

cache_path = os.path.join(basedir, f'signal_frames/event_0/signals.cache')
if not(os.path.exists(cache_path)):
    cache_path = os.path.join(basedir, f'combined_frames/event_0/signals.cache')
ifos = ast.literal_eval(config.get('analysis','ifos'))
PSD_paths = {}
channels = {}
for ifo in ifos:
    #PSD_paths[ifo] = os.path.join(rundir, 'C1_at_H1_fromascii_psd.xml.gz')
    PSD_paths[ifo] = os.path.join(rundir, f'{ifo}-psd.xml.gz')
    channels[ifo] = f'{ifo}:FAKE-STRAIN'
    
# segment length
seglen = int(config.get('engine','seglen'))

# harmonic setting
lmax = int(config.get('rift-pseudo-pipe', 'l-max'))
    
## Determine deltaF from PSDs
# Note that for the PSDs we're currently using, deltaF = 0.0625.
deltaF = {}
for ifo in ifos:
    #deltaF[ifo] = lalsimutils.get_psd_series_from_xmldoc(PSD_paths[ifo], ifo).deltaF
    #deltaF[ifo] = 0.031250 # trying to force this for now...
    if seglen == 32:
        deltaF[ifo] = 0.031250
    elif seglen == 64:
        deltaF[ifo] = 0.015625
    elif seglen == 128:
        deltaF[ifo] = 0.0078125
        
    
print(f'deltaF from inject_compare is: {deltaF}')
    
## Load injection frames here ##
frame_files = {}
with open(cache_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    gwf_path_raw = line.split()[-1]
    gwf_path = gwf_path_raw.replace('file://localhost', '')
    ifo_name = gwf_path.split('/')[-1][0] + '1'

    frame_files[ifo_name] = gwf_path # these are paths to the frame files
    

    
    
dist_marg_path = os.path.join(rundir, 'distance_marginalization_lookup.npz') # will probably need to put in a try/except for this




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

def gen_wav(P_dict, name):
    # name is e.g. truth, maxL, test
    wav_data = {}
    # each P_dict has a P object per ifo
    for ifo in P_dict.keys():
        print(f'Now doing gen_wav for {ifo}')
        
        hlmT = {}
        hlmF, hlmF_conj = lalsimutils.std_and_conj_hlmoff(P_dict[ifo], Lmax=lmax) # modes in fourier space
        for mode in hlmF:
            hlmT[mode] = lalsimutils.DataInverseFourier(hlmF[mode]) # convert to time series
            
        hoft = lalsimutils.hoft_from_hlm(hlmT, P_dict[ifo], return_complex=False) # combine modes
        
        #print(f'print hoft.epoch from inject_compare: {hoft.epoch}')
        
        wav_data[ifo] = {}
        
        wav_data[ifo]['hoft'] = hoft
        wav_data[ifo]['hlmoft'] = hlmT
        
        hoft_frames = {}
        # Now project hoft to ifo frames
        hoft_frames[ifo] = lalframe.FrameNew(wav_data[ifo]['hoft'].epoch, wav_data[ifo]['hoft'].deltaT*wav_data[ifo]['hoft'].data.length, "LIGO", 0,0,0)
        wav_data[ifo]['hoft'].name = channels[ifo]
        lalframe.FrameAddREAL8TimeSeriesProcData(hoft_frames[ifo], wav_data[ifo]['hoft'])
        frame_path = {}
        frame_path[ifo] = {}
        frame_path[ifo] = os.path.join(diag_path, f'{name}_frame_{ifo}')
        lalframe.FrameWrite(hoft_frames[ifo], frame_path[ifo])

        frame_data = {}      
        frame_data[ifo] = read_frame(frame_path[ifo], channels[ifo])        

        wav_data[ifo]['strain'] = frame_data[ifo]['strain']
        wav_data[ifo]['sample_times'] = frame_data[ifo]['sample_times']
            
        
    return wav_data

  
def run_ILE(xml, output_name):
    # extract ILE settings from command-single
    cs_path = os.path.join(rundir, 'command-single.sh')
    
    with open(cs_path, 'r') as file:
        lines = file.readlines()
    
    # Skip the first two lines and get the options line
    options_line = lines[2].strip()
    parts = options_line.split()
    
    # Dictionary of new values for specific options
    new_values = {
        '--cache': cache_path,
        '--distance-marginalization-lookup-table': dist_marg_path,
        '--sim-xml': xml,
        '--n-events-to-analyze': '1',
        '--l-max': lmax
    }
    
    # Add PSD paths for the relevant IFOs
    for ifo in ifos:
        new_values[f'--psd-file {ifo}='] = PSD_paths[ifo]
    
    if args.fast_ILE:
        new_values['--n-max'] = 20000
        
    # process and clean options
    options = []
    seen_options = set()
    i = 0
    
    
    
    # Split the line into parts, keeping paired options together
    parts = options_line.split()
    options = []
    i = 0  # Start from the first element

    while i < len(parts):
        if parts[i].startswith('--'):
            if i + 1 < len(parts) and not parts[i + 1].startswith('--'):
                option = parts[i]
                value = parts[i + 1]

                # Update value if needed
                if option in new_values:
                    value = new_values[option]

                options.append(f'{option} {value}')
                i += 2  # Move past option and value
            else:
                # Handle options without a value
                option = parts[i]
                if option in new_values:
                    options.append(f'{option} {new_values[option]}')
                else:
                    options.append(option)
                i += 1  # Move past single option
        else:
            i += 1  # Increment for non-option elements
    
    # Add new options
    options.append('--force-xpy')
    output_path = os.path.join(diag_path, f'{output_name}.xml')
    options.append(f'--output-file {output_path}')
    
    if args.pin_extrinsics:
        Q = lalsimutils.xml_to_ChooseWaveformParams_array(xml)[0]
        true_dec = Q.theta
        true_ra = Q.phi
        true_tref = Q.tref
        options.append(f' --declination {true_dec} ')
        options.append(f' --right-ascension {true_ra} ')
        options.append(f' --t-ref {true_tref} ')
        
    if args.pin_distance:
        Pd = lalsimutils.xml_to_ChooseWaveformParams_array(xml)[0]
        true_dist = Pd.dist
        options.append(f' --distance {true_dist} ')
        options.remove('--distance-marginalization')
        
    #if args.test_AV:
    #    options.append('--sampler-method AV')
        
    if args.verbose_ILE:
        options.append('--verbose')
    
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
    lnL_value = np.loadtxt(output_path + '_0_.dat')[11]

    print(f'The lnL value is {lnL_value}')    

    return lnL_value
    
    

##############################
# Injected Signal
##############################
print('###################################')
print('Now generating true param waveforms')
print('###################################')

# These are the injected parameters
truth_xml = os.path.join(basedir, 'mdc.xml.gz')

# Save ChooseWaveformParams object for later use - per ifo
P_truth = {}
for ifo in ifos: 
    P_truth[ifo] = lalsimutils.xml_to_ChooseWaveformParams_array(truth_xml)[0]
    P_truth[ifo].deltaT = 1./srate
    P_truth[ifo].deltaF = deltaF[ifo]
    P_truth[ifo].detector = ifo
    P_truth[ifo].radec = True
       
        
# Generate waveforms
#
wav_data_truth = gen_wav(P_truth, 'truth')


##############################
# User test waveform
##############################

if args.user_test:
    print('###################################')
    print('Now generating test param waveforms')
    print('###################################')
    
    P_test = {}
    for ifo in ifos:
        # Load injected params into lalsimutils
        P_test[ifo] = P_truth[ifo].copy()
        
        # override intrinsics with test values
        P_test[ifo].m1 = 6.054219999999999935*SM
        P_test[ifo].m2 = 5.786400000000000432*SM
        P_test[ifo].s1x = 0.0
        P_test[ifo].s1y = 0.0
        P_test[ifo].s1z = 0.0
        P_test[ifo].s2x = 0.0
        P_test[ifo].s2y = 0.0
        P_test[ifo].s2z = 0.0
        P_test[ifo].E0 = 1.044620000000000104
        P_test[ifo].p_phi0 =  9.541669999999999874
            
        # dump test params to xml - only need to do once - H1 is the P default
        if ifo == 'H1':
            lalsimutils.ChooseWaveformParams_array_to_xml([P_test[ifo]], fname=os.path.join(diag_path,'user_test'))


    # Generate waveforms
    wav_data_test = gen_wav(P_test, 'test')
        
    # specify xml path as a variable for ILE calc
    test_xml = os.path.join(diag_path, 'user_test.xml.gz')
    
##############################
# Max Likelihood Waveform
##############################

if args.maxL:
    print('###################################')
    print('Now generating maxL param waveform')
    print('###################################')
    
    # These are the evaluated grid points
    comp_file = os.path.join(rundir, 'all.net')

    max_L = np.loadtxt(comp_file)[0] # load top row of all.net
    maxL_lnL_fromfile = max_L[11]

    P_maxL = P_truth.copy() # copy all settings from the truth file
    
    P_maxL = {}
    for ifo in ifos:
        # Load injected params into lalsimutils
        P_maxL[ifo] = P_truth[ifo].copy()

        # override intrinsics with max_L values

        P_maxL[ifo].m1 = max_L[1]*SM
        P_maxL[ifo].m2 = max_L[2]*SM
        P_maxL[ifo].s1x = max_L[3]
        P_maxL[ifo].s1y = max_L[4]
        P_maxL[ifo].s1z = max_L[5]
        P_maxL[ifo].s2x = max_L[6]
        P_maxL[ifo].s2y = max_L[7]
        P_maxL[ifo].s2z = max_L[8]
        P_maxL[ifo].E0 = max_L[9]
        P_maxL[ifo].p_phi0 = max_L[10]
            
        # dump test params to xml - only need to do once - H1 is the P default
        if ifo == 'H1':
            lalsimutils.ChooseWaveformParams_array_to_xml([P_maxL[ifo]], fname=os.path.join(diag_path,'maxL'))
    
    # Generate waveforms
    wav_data_maxL = gen_wav(P_maxL, 'maxL')   
    
    # specify xml path as a variable for ILE calc
    maxL_xml = os.path.join(diag_path, 'maxL.xml.gz')
    
####################################
# Top N lnL Waveforms - no ILE calc
####################################
    
if args.maxL and args.plot_topN > 0:
    print(f'###################################')
    print(f'Generating waveforms for top {args.plot_topN} rows (excluding maxL) in composite file')
    print(f'###################################')

    comp_file = os.path.join(rundir, 'all.net')
    top_N = np.loadtxt(comp_file)[1:args.plot_topN + 1]  # Skip top row (maxL) and get next N rows
    P_topN = [{} for _ in range(args.plot_topN)]
    wav_data_topN = []  # List to store waveform data for each top-N row
    lnL_values_topN = []  # List to store lnL values for each top-N row

    for i, row in enumerate(top_N):
        lnL_values_topN.append(row[11])  # Extract lnL value (index 11)
        for ifo in ifos:
            P_topN[i][ifo] = P_truth[ifo].copy()

            # Override params with values from the row
            P_topN[i][ifo].m1 = row[1] * SM
            P_topN[i][ifo].m2 = row[2] * SM
            P_topN[i][ifo].s1x = row[3]
            P_topN[i][ifo].s1y = row[4]
            P_topN[i][ifo].s1z = row[5]
            P_topN[i][ifo].s2x = row[6]
            P_topN[i][ifo].s2y = row[7]
            P_topN[i][ifo].s2z = row[8]
            P_topN[i][ifo].E0 = row[9]
            P_topN[i][ifo].p_phi0 = row[10]

        # Generate and store the waveform data
        wav_data_topN.append(gen_wav(P_topN[i], f'topN_{i+1}'))




##############################
# ILE calculation
##############################

#new_data_start = 999999994.0
#new_data_end = 1000000002.0
#new_seglen = int(new_data_end - new_data_start)

if args.calc_ILE:
    print('###################################')
    print('Now running ILE on true params')
    print('###################################')
    truth_lnL = run_ILE(truth_xml, 'truth_output')
    print(f'The lnL value for the true params is {truth_lnL}')
    
    if args.user_test:
        print('###################################')
        print('Now running ILE on test params')
        print('###################################')
        test_lnL = run_ILE(test_xml, 'test_output')
        print(f'The lnL value for the test params is {test_lnL}')
    
    if args.maxL:
        print('###################################')
        print('Now running ILE on maxL params')
        print('###################################')
        maxL_lnL = run_ILE(maxL_xml, 'maxL_output')
        print(f'The lnL value for the maxL params is {maxL_lnL}')

    
    if args.maxL and args.user_test:        
        result_list = [truth_lnL, test_lnL, maxL_lnL]
        result_file = os.path.join(diag_path, f'Injection+maxL+test_mtot-{(P_test["H1"].m1+P_test["H1"].m2)/SM}_q-{P_test["H1"].m1/P_test["H1"].m2}_chi1-{P_test["H1"].s1z}_chi2-{P_test["H1"].s2z}_E0-{P_test["H1"].E0}_pphi0-{P_test["H1"].p_phi0}_lmax-{lmax}_ILEresults.txt')
                                       
    elif args.maxL and not(args.user_test):
        result_list = [truth_lnL, maxL_lnL]
        result_file = os.path.join(diag_path, f'Injection+maxL_lmax-{lmax}_ILEresults.txt')
        
    elif args.user_test and not(args.maxL):
        result_list = [truth_lnL, test_lnL]
        result_file = os.path.join(diag_path, f'Injection+test_mtot-{(P_test["H1"].m1+P_test["H1"].m2)/SM}_q-{P_test["H1"].m1/P_test["H1"].m2}_chi1-{P_test["H1"].s1z}_chi2-{P_test["H1"].s2z}_E0-{P_test["H1"].E0}_pphi0-{P_test["H1"].p_phi0}_lmax-{lmax}_ILEresults.txt')
    else:
        result_list = [truth_lnL]
        result_file = os.path.join(diag_path, f'Injection_lmax-{lmax}_ILEresults.txt')

    # Open the file in write mode
    with open(result_file, "w") as file:
        # Write each item in the list to the file
        for item in result_list:
            file.write(f"{item}\n")


##############################
# Plotting
##############################

if not(args.no_plots):

    # read injected frame data into variables
    injected_frame_data = {}
    amps = []
    for ifo in ifos:
        injected_frame_data[ifo] = read_frame(frame_files[ifo], channels[ifo])
        
        # check strain for plot limits
        amps.append(np.amax(np.abs(injected_frame_data[ifo]['strain'])))
        
    #ylim = max(injected_amp)
       
    


    # Create plots per ifo. Note that the lnL values will be the same on each one

    for ifo in ifos:
        print(f'Now plotting for {ifo}')

        fig, axs = plt.subplots(figsize=(20, 10), nrows=2, ncols=1)
        ax = axs.flatten()

        ## All plots ##

        # frame plot
        ax[0].plot(injected_frame_data[ifo]['sample_times'], injected_frame_data[ifo]['strain'], color='orange', alpha=1.0,  label=f'{channels[ifo]}')

        ax[1].plot(injected_frame_data[ifo]['sample_times'], injected_frame_data[ifo]['strain'], color='orange', alpha=1.0,  label=f'{channels[ifo]}')
            

        # True value plot               
        ax[0].plot(wav_data_truth[ifo]['sample_times'], wav_data_truth[ifo]['strain'], color='blue', linestyle='dashed', label=f'True params')

        ax[1].plot(wav_data_truth[ifo]['sample_times'], wav_data_truth[ifo]['strain'], color='blue', linestyle='dashed', label=f'True params')

        amps.append(np.amax(np.abs(wav_data_truth[ifo]['strain'])))
                
        

        if args.user_test:
            # Test value plot    
            ax[0].plot(wav_data_test[ifo]['sample_times'], wav_data_test[ifo]['strain'], color='green', linestyle='dotted', label=f'Test params')

            ax[1].plot(wav_data_test[ifo]['sample_times'], wav_data_test[ifo]['strain'], color='green', linestyle='dotted', label=f'Test params')
            
            amps.append(np.amax(np.abs(wav_data_test[ifo]['strain'])))

        if args.maxL:
            maxL_label = f'maxL lnL={maxL_lnL:.2f}' if args.calc_ILE else 'maxL params'
            # Test value plot    
            ax[0].plot(wav_data_maxL[ifo]['sample_times'], wav_data_maxL[ifo]['strain'], color='red', linestyle='dashdot', label=maxL_label)

            ax[1].plot(wav_data_maxL[ifo]['sample_times'], wav_data_maxL[ifo]['strain'], color='red', linestyle='dashdot', label=maxL_label)
            
            amps.append(np.amax(np.abs(wav_data_maxL[ifo]['strain'])))
            
        if args.plot_topN > 0:
            for i, data in enumerate(wav_data_topN):  # Iterate through stored wave data
                alpha_value = 0.8 - i * (0.8 - 0.2) / max(1, (args.plot_topN - 1)) # decrease opacity as f(lnL)
                lnL_label = f'Top {i+2} lnL={lnL_values_topN[i]:.2f}' if args.calc_ILE else f'Top {i+2} params'
                # Plot only the current IFO
                ax[0].plot(
                    data[ifo]['sample_times'], 
                    data[ifo]['strain'], 
                    color='red', 
                    alpha=alpha_value, 
                    linestyle='dashdot', 
                    label=lnL_label
                )
                ax[1].plot(
                    data[ifo]['sample_times'], 
                    data[ifo]['strain'], 
                    color='red', 
                    alpha=alpha_value, 
                    linestyle='dashdot', 
                    label=lnL_label
                )
                
                amps.append(np.amax(np.abs(data[ifo]['strain'])))


        ## Plot annotations and settings

        ax[0].grid(True, alpha=0.4)
        ax[1].set_xlabel('Time [s]')
        ax[0].set_ylabel('Strain')
        ax[1].set_ylabel('Strain')
        #ax[0].set_xlim(-0.05 + 1e9, 0.05 + 1e9)
        ax[0].set_xlim(-1.0 + 1e9, 1.0 + 1e9)
        
        ylim = max(amps)
        
        ax[0].set_ylim(-1.0*ylim, 1.0*ylim)
        ax[1].set_ylim(-1.0*ylim, 1.0*ylim)
        #if args.frame_plot:
            #ax[0].set_xlim(-0.25 + 1e9, 0.25 + 1e9)
        #else:
            #ax[0].set_xlim(-0.25, 0.25)
        ax[1].legend(loc='upper right')
        ax[0].set_title('Zoomed in')
        ax[1].set_xlim(-12.0 + 1e9, 20.0 + 1e9) # for now
            #if args.frame_plot:
            #    ax[1].set_xlim(-10.0 + 1e9, 10.0 + 1e9) # for now
            #else:
            #    ax[1].set_xlim(-10.0, 10.0)
        ax[1].set_title('Taper domain')

        
        textstr_inj = '\n'.join((
            'Injected:',
            '$M = %.1f \: [M_{\odot}]$' % ((P_truth[ifo].m1 + P_truth[ifo].m2)/SM),
            '$q = %.3f $' % (P_truth[ifo].m1/P_truth[ifo].m2),
            '$S_{1,z}=%.2f$' % (P_truth[ifo].s1z),
            '$S_{2,z}=%.2f$' % (P_truth[ifo].s2z),
            '$E_0=%.5f$' % (P_truth[ifo].E0),
            '$p_{\phi}^0=%.5f$' % (P_truth[ifo].p_phi0)))

        ax[0].text(0.0200, 0.98, textstr_inj, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if args.user_test:
            textstr_test = '\n'.join((
                'Test waveform:',
                '$M = %.1f \: [M_{\odot}]$' % ((P_test[ifo].m1 + P_test[ifo].m2)/SM),
                '$q = %.3f $' % (P_test[ifo].m1/P_test[ifo].m2),
                '$S_{1,z}=%.2f$' % (P_test[ifo].s1z),
                '$S_{2,z}=%.2f$' % (P_test[ifo].s2z),
                '$E_0=%.5f$' % (P_test[ifo].E0),
                '$p_{\phi}^0=%.5f$' % (P_test[ifo].p_phi0)))

            ax[0].text(0.2200, 0.98, textstr_test, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if args.maxL:
            textstr_maxL = '\n'.join((
                'Max Likelihood:',
                '$M = %.1f \: [M_{\odot}]$' % ((P_maxL[ifo].m1 + P_maxL[ifo].m2)/SM),
                '$q = %.3f $' % (P_maxL[ifo].m1/P_maxL[ifo].m2),
                '$S_{1,z}=%.2f$' % (P_maxL[ifo].s1z),
                '$S_{2,z}=%.2f$' % (P_maxL[ifo].s2z),
                '$E_0=%.5f$' % (P_maxL[ifo].E0),
                '$p_{\phi}^0=%.5f$' % (P_maxL[ifo].p_phi0)))

            ax[0].text(0.1200, 0.98, textstr_maxL, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        lmax_str = f'lmax = {lmax}' 

        ax[0].text(0.01, 0.01, lmax_str, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        try:
            if args.maxL and args.user_test:
                L_str = '\n'.join((
            'Calculated here:',
            f'Truth lnL value: {truth_lnL}',
            f'Test lnL value: {test_lnL}',
            f'maxL lnL value: {maxL_lnL}'))
            elif args.maxL and not(args.user_test):
                L_str = '\n'.join((
            'Calculated here:',
            f'Truth lnL value: {truth_lnL}',
            f'maxL lnL value: {maxL_lnL}'))
            elif args.user_test and not(args.maxL):
                L_str = '\n'.join((
            'Calculated here:',
            f'Truth lnL value: {truth_lnL}',
            f'Test lnL value: {test_lnL}'))
            else:
                L_str = f'Truth lnL value: {truth_lnL}'

            ax[1].text(0.01, 0.01, L_str, transform=ax[1].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except:
            print('No Likelihood values')


        # Savepath variations
        if args.calc_ILE:
            if args.maxL and args.user_test:
                savepath = os.path.join(
                    diag_path,
                    f"Injection+maxL+test_mtot-{(P_test[ifo].m1 + P_test[ifo].m2) / SM}_q-{P_test[ifo].m1 / P_test[ifo].m2}_chi1-{P_test[ifo].s1z}_chi2-{P_test[ifo].s2z}_E0-{P_test[ifo].E0}_pphi0-{P_test[ifo].p_phi0}_lmax-{lmax}_ILEresults_{ifo}-frame.png"
                )
            elif args.maxL and not args.user_test:
                savepath = os.path.join(
                    diag_path,
                    f"Injection+maxL_lmax-{lmax}_ILEresults_{ifo}-frame.png"
                )
            elif args.user_test and not args.maxL:
                savepath = os.path.join(
                    diag_path,
                    f"Injection+test_mtot-{(P_test[ifo].m1 + P_test[ifo].m2) / SM}_q-{P_test[ifo].m1 / P_test[ifo].m2}_chi1-{P_test[ifo].s1z}_chi2-{P_test[ifo].s2z}_E0-{P_test[ifo].E0}_pphi0-{P_test[ifo].p_phi0}_lmax-{lmax}_ILEresults_{ifo}-frame.png"
                )
            else:
                savepath = os.path.join(
                    diag_path, f"Injection_lmax-{lmax}_ILEresults_{ifo}-frame.png"
                )
                
        else:
            if args.maxL and args.user_test:
                savepath = os.path.join(
                    diag_path,
                    f"Injection+maxL+test_mtot-{(P_test[ifo].m1 + P_test[ifo].m2) / SM}_q-{P_test[ifo].m1 / P_test[ifo].m2}_chi1-{P_test[ifo].s1z}_chi2-{P_test[ifo].s2z}_E0-{P_test[ifo].E0}_pphi0-{P_test[ifo].p_phi0}_lmax-{lmax}_NoILE_{ifo}-frame.png"
                )
            elif args.maxL and not args.user_test:
                savepath = os.path.join(
                    diag_path,
                    f"Injection+maxL_lmax-{lmax}_NoILE_{ifo}-frame.png"
                )
            elif args.user_test and not args.maxL:
                savepath = os.path.join(
                    diag_path,
                    f"Injection+test_mtot-{(P_test[ifo].m1 + P_test[ifo].m2) / SM}_q-{P_test[ifo].m1 / P_test[ifo].m2}_chi1-{P_test[ifo].s1z}_chi2-{P_test[ifo].s2z}_E0-{P_test[ifo].E0}_pphi0-{P_test[ifo].p_phi0}_lmax-{lmax}_NoILE_{ifo}-frame.png"
                )
            else:
                savepath = os.path.join(
                    diag_path, f"Injection_lmax-{lmax}_NoILE_{ifo}-frame.png"
                )
            

        print(f'Now saving to {savepath}')
        
        #print(ax[0].get_xlim(), ax[0].get_ylim())
        #print(ax[1].get_xlim(), ax[1].get_ylim())
        
        plt.savefig(savepath, bbox_inches='tight')

else:
    print('No plots enabled, program is complete.')