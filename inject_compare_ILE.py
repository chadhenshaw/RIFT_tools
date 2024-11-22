#!/usr/bin/env python3

# Comparing likelihood values for an injection

#import RIFT.lalsimutils_test as lalsimutils
import RIFT.lalsimutils as lalsimutils
import RIFT.likelihood.factored_likelihood as factored_likelihood  # direct hoft call
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
    parser.add_argument("--event", type=int, default=0, help='Event number. ADVANCED USERS ONLY')
    parser.add_argument("--no-plots", action='store_true', help='Does not generate or plot waveforms - for use with --calc-ILE')
    parser.add_argument("--test-AV", action='store_true', help='Uses the AV sampler for ILE')
    parser.add_argument('--skip-truth-ILE',action='store_true', help='Skips calculating ILE for the true params')
    parser.add_argument('--check-frame-duplicate', action='store_true', help='Recreates frame data just like util_LALWriteFrame')
    parser.add_argument('--plot-topN', type=int, default=0, help='Plot waveforms for the top N rows (excluding maxL) from all.net. Requires --maxL.')
    parser.add_argument('--taper-domain',action='store_true', help='Shows the taper domain instead of the full frame duration')
    parser.add_argument('--force-single-IFO', action='store_true', help='Runs only with L1')
    #parser.add_argument('--make-L1-frame', action='store_true', help='Makes a new frame with L1 ideal sky location')
    #parser.add_argument('--make-22-frame', action='store_true', help='Makes new frames with just 22 modes')
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

### Generating new frames with single_inject ###

#if args.make_L1_frame:
    # copy config file
#    new_config = os.path.join(diag_path, "updated_config_L1-ideal.ini")
#    shutil.copy2(config_file, new_config)
    
    # Load the copied config file
#    config.read(new_config)
    
    # Update the working_directory
#    current_working_dir = config.get("init", "working_directory")
#    updated_working_dir = f"L1-ideal_{current_working_dir}"
#    config.set("init", "working_directory", updated_working_dir)
    
    # Update the rundir
#    config.set("init", "rundir", 'analysis_0_L1-ideal')
    
    # Update sky location
#    config.set("injection-parameters", "dec", "0.533")
#    config.set("injection-parameters", "ra", "5.17")
    
    # Write the updated config back to the file
#    with open(new_config, "w") as configfile:
#        config.write(configfile)
    
    # run single inject to make new frames
#    os.chdir(diag_path)
    
    # run single inject
#    single_inject_exe = os.path.join(os.path.dirname(__file__), 'single_inject_RIFT.py')
    
#    cmd = f'python {single_inject_exe} --use-ini {new_config} --use-hyperbolic --add-extrinsic --use-osg --hlmoft-frames --just-frames '
    
#    print(cmd)
#    os.system(cmd)
    
    # get new cache path
#    new_cache_path = os.path.join(os.path.join(diag_path, updated_working_dir), 'combined_frames/event_0/signals.cache')


# Necessary paths for ILE calculation
cache_path = os.path.join(basedir, f'combined_frames/event_{args.event}/signals.cache')
if args.force_single_IFO:
    print("Forcing sky location to L1, running only with L1 detector")
    ifos = ['L1']
else:
    ifos = ast.literal_eval(config.get('analysis','ifos'))
PSD_paths = {}
channels = {}
for ifo in ifos:
    PSD_paths[ifo] = os.path.join(rundir, f'{ifo}-psd.xml.gz')
    channels[ifo] = f'{ifo}:FAKE-STRAIN'
    
## Determine deltaF from PSDs
# Note that for the PSDs we're currently using, deltaF = 0.0625.
deltaF = {}
for ifo in ifos:
    deltaF[ifo] = lalsimutils.get_psd_series_from_xmldoc(PSD_paths[ifo], ifo).deltaF
    
## Load injection frames here ##
frame_files = {}
with open(cache_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    gwf_path_raw = line.split()[-1]
    gwf_path = gwf_path_raw.replace('file://localhost', '')
    ifo_name = gwf_path.split('/')[-1][0] + '1'

    frame_files[ifo_name] = gwf_path # these are paths to the frame files
    

    
    
dist_marg_path = os.path.join(basedir, 'distance_marginalization_lookup.npz') # will probably need to put in a try/except for this




##############################
# Function definitions
##############################

def read_frame(frame_file, channel):
    frame_data = {}
    raw_data = TimeSeries.read(frame_file, channel)
    frame_data['strain'] = np.array(raw_data.value)
    frame_data['sample_times'] = np.array(raw_data.times)
        
    return frame_data
    

def gen_wav(P_dict, name):
    # name is e.g. truth, maxL, test
    wav_data = {}
    # each P_dict has a P object per ifo
    for ifo in P_dict.keys():
        print(f'Now doing gen_wav for {ifo}')
        
        #P_dict[key].print_params()
        if args.hlmoff:
            hlmT = {}
            hlmF, hlmF_conj = lalsimutils.std_and_conj_hlmoff(P_dict[ifo], Lmax=args.lmax) # modes in fourier space
            for mode in hlmF:
                hlmT[mode] = lalsimutils.DataInverseFourier(hlmF[mode]) # convert to time series
        else:
            hlmT = lalsimutils.hlmoft(P_dict[ifo], Lmax=args.lmax)
            
        hoft = lalsimutils.hoft_from_hlm(hlmT, P_dict[ifo], return_complex=False) # combine modes
        
        wav_data[ifo] = {}
        
        wav_data[ifo]['hoft'] = hoft
        
        
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

def gen_wav_o4c(P_dict, name):
    # IN DEVELOPMENT
    # NOTE: only works in the O4c branch
    # name is e.g. truth, maxL, test
    wav_data = {}
    # each P_dict has a P object per ifo
    for ifo in P_dict.keys():
        print(f'Now doing gen_wav for {ifo}')
        
        P_copy = P_dict[ifo].manual_copy()
        
        if args.check_frame_duplicate:
            # Same procedure that LALWriteFrame uses
            T_est = lalsimutils.estimateWaveformDuration(P_copy)
            T_est = P_copy.deltaT*lalsimutils.nextPow2(T_est/P_copy.deltaT)
            P_copy.deltaF = 1./T_est
            print(f'P.deltaF is: {P_copy.deltaF}')
            hoft = lalsimutils.hoft(P_copy)
        else:
            #extra_args = {}
            extra_waveform_args = {}
            extra_waveform_args['fd_centering_factor']= 0.9
            
            # testing forced P.deltaF = 0.03125
            P_copy.deltaF = 0.03125
            
            hlmF_1, _= factored_likelihood.internal_hlm_generator(P_copy, args.lmax, extra_waveform_kwargs=extra_waveform_args)
            hlmT_1 = {}
            for mode in hlmF_1:
                hlmT_1[mode] = lalsimutils.DataInverseFourier(hlmF_1[mode])

            P_copy = P_dict[ifo].manual_copy()
            hoft = lalsimutils.hoft_from_hlm(hlmT_1, P_copy,return_complex=False)
        
        wav_data[ifo] = {}
        
        wav_data[ifo]['hoft'] = hoft
        
        
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
        '--l-max': args.lmax
    }
    
    # Add PSD paths for the relevant IFOs
    if args.force_single_IFO:
        new_values[f'--psd-file L1='] = PSD_paths['L1']
    else:
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

                # Handle specific options like --psd-file or --channel-name
                if option == '--psd-file' and args.force_single_IFO:
                    # Skip options for H1 and V1 if forcing single IFO
                    if 'H1=' in value or 'V1=' in value:
                        i += 2  # Skip both the option and its value
                        continue
                elif option == '--channel-name' and args.force_single_IFO:
                    # Skip --channel-name for H1 and V1
                    if 'H1=' in value or 'V1=' in value:
                        i += 2  # Skip both the option and its value
                        continue
                elif option == '--fmin-ifo' and args.force_single_IFO:
                    # Skip --fmin-ifo for H1 and V1
                    if 'H1=' in value or 'V1=' in value:
                        i += 2  # Skip both the option and its value
                        continue

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
    if args.test_AV:
        options.append('--sampler-method AV')
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
    if not(args.use_hyperbolic):
        P_truth[ifo].taper=lalsimutils.lsu_TAPER_START
    #P_truth[ifo].print_params()
        
        
if not(args.no_plots):
    # Generate waveforms
    #
    if args.use_hyperbolic:
        wav_data_truth = gen_wav(P_truth, 'truth')
    else:
        wav_data_truth = gen_wav_o4c(P_truth, 'truth')


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
        P_test[ifo].m1 = 80.0*SM
        P_test[ifo].m2 = 20.0*SM
        P_test[ifo].s1x = 0.0
        P_test[ifo].s1y = 0.0
        P_test[ifo].s1z = 0.0
        P_test[ifo].s2x = 0.0
        P_test[ifo].s2y = 0.0
        P_test[ifo].s2z = 0.0
        if args.use_hyperbolic:
            P_test[ifo].E0 = 1.024
            P_test[ifo].p_phi0 =  5.2
            
        # dump test params to xml - only need to do once - H1 is the P default
        if ifo == 'L1':
            lalsimutils.ChooseWaveformParams_array_to_xml([P_test[ifo]], fname=os.path.join(diag_path,'user_test'))

    if not(args.no_plots):
        # Generate waveforms
        wav_data_test = gen_wav(P_test, 'test')
        
    # specify xml path as a variable for ILE calc
    test_xml = os.path.join(diag_path, 'user_test.xml.gz')
    
if args.test_from_log:
    args.user_test = True # so that subsequent syntax works
    print('###################################')
    print('Now generating log test param waveforms')
    print('###################################')
    # try to load log file
    log_path = os.path.join(args.analysis_dir, 'diagnostic/exclusions_log_test.txt')
    
    try:
        if not os.path.isfile(log_path):
            try:
                # updated file name in newer code
                log_path = os.path.join(args.analysis_dir, 'diagnostic/SigmaOverL_exclusions.txt')
                if not os.path.isfile(log_path):
                    raise FileNotFoundError(f"No exlcusions log found! Skipping!")
            except FileNotFoundError as e:
                print(e)  # Handle error if needed
    
        data = np.loadtxt(log_path)
        rand_line = data[np.random.choice(data.shape[0])]

        # Load injected params into lalsimutils
        P_test = {}
        for ifo in ifos:
            # Load injected params into lalsimutils
            P_test[ifo] = P_truth[ifo].copy()

            # assign values from log to lalsimutils
            P_test[ifo].m1 = float(rand_line[1])*SM
            P_test[ifo].m2 = float(rand_line[2])*SM
            P_test[ifo].s1x = float(rand_line[3])
            P_test[ifo].s1y = float(rand_line[4])
            P_test[ifo].s1z = float(rand_line[5])
            P_test[ifo].s2x = float(rand_line[6])
            P_test[ifo].s2y = float(rand_line[7])
            P_test[ifo].s2z = float(rand_line[8])
            if args.use_hyperbolic:
                P_test[ifo].E0 = float(rand_line[9]) # NEEDS TO BE TESTED!!
                P_test[ifo].p_phi0 = float(rand_line[10]) # NEEDS TO BE TESTED!!
                
            # dump test params to xml - only need to do once - H1 is the P default
            if ifo == 'L1':
                lalsimutils.ChooseWaveformParams_array_to_xml([P_test[ifo]], fname=os.path.join(diag_path,'user_test'))
        
        if not(args.no_plots):
            # Generate waveforms
            wav_data_test = gen_wav(P_test, 'test')
        
        # specify xml path as a variable for ILE calc
        test_xml = os.path.join(diag_path, 'user_test.xml.gz')
        
    except FileNotFoundError as e:
        print(e)
        pass

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
    if args.use_hyperbolic:
        maxL_lnL_fromfile = max_L[11]
    else:
        maxL_lnL_fromfile = max_L[9]

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
        if args.use_hyperbolic:
            P_maxL[ifo].E0 = max_L[9]
            P_maxL[ifo].p_phi0 = max_L[10]
            
        # dump test params to xml - only need to do once - H1 is the P default
        if ifo == 'L1':
            lalsimutils.ChooseWaveformParams_array_to_xml([P_maxL[ifo]], fname=os.path.join(diag_path,'maxL'))
    
    if not(args.no_plots):
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
        if args.use_hyperbolic:
            lnL_values_topN.append(row[11])  # Extract lnL value (index 11)
        else:
            lnL_values_topN.append(row[9])  # Extract lnL value (index 9)
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
            if args.use_hyperbolic:
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
    if not(args.skip_truth_ILE):
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
        if args.use_hyperbolic:
            result_file = os.path.join(diag_path, f'Injection+maxL+test_mtot-{(P_test["L1"].m1+P_test["L1"].m2)/SM}_q-{P_test["L1"].m1/P_test["L1"].m2}_chi1-{P_test["L1"].s1z}_chi2-{P_test["L1"].s2z}_E0-{P_test["L1"].E0}_pphi0-{P_test["L1"].p_phi0}_lmax-{args.lmax}_ILEresults.txt')
        else:
            result_file = os.path.join(diag_path, f'Injection+maxL+test_mtot-{(P_test["L1"].m1+P_test["L1"].m2)/SM}_q-{P_test["L1"].m1/P_test["L1"].m2}_chi1-{P_test["L1"].s1z}_chi2-{P_test["L1"].s2z}_lmax-{args.lmax}_ILEresults.txt')
                                       
    elif args.maxL and not(args.user_test):
        result_list = [truth_lnL, maxL_lnL]
        result_file = os.path.join(diag_path, f'Injection+maxL_lmax-{args.lmax}_ILEresults.txt')
        
    elif args.user_test and not(args.maxL):
        result_list = [truth_lnL, test_lnL]
        if args.use_hyperbolic:
            result_file = os.path.join(diag_path, f'Injection+test_mtot-{(P_test["L1"].m1+P_test["L1"].m2)/SM}_q-{P_test["L1"].m1/P_test["L1"].m2}_chi1-{P_test["L1"].s1z}_chi2-{P_test["L1"].s2z}_E0-{P_test["L1"].E0}_pphi0-{P_test["L1"].p_phi0}_lmax-{args.lmax}_ILEresults.txt')
        else:
            result_file = os.path.join(diag_path, f'Injection+test_mtot-{(P_test["L1"].m1+P_test["L1"].m2)/SM}_q-{P_test["L1"].m1/P_test["L1"].m2}_chi1-{P_test["L1"].s1z}_chi2-{P_test["L1"].s2z}_lmax-{args.lmax}_ILEresults.txt')
    else:
        if not(args.skip_truth_ILE):
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

if not(args.no_plots):

    # read injected frame data into variables
    injected_frame_data = {}
    for ifo in ifos:
        injected_frame_data[ifo] = read_frame(frame_files[ifo], channels[ifo])


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

        if args.user_test:
            # Test value plot    
            ax[0].plot(wav_data_test[ifo]['sample_times'], wav_data_test[ifo]['strain'], color='green', linestyle='dotted', label=f'Test params')

            ax[1].plot(wav_data_test[ifo]['sample_times'], wav_data_test[ifo]['strain'], color='green', linestyle='dotted', label=f'Test params')

        if args.maxL:
            maxL_label = f'maxL lnL={maxL_lnL_fromfile:.2f}' if args.calc_ILE else 'maxL params'
            # Test value plot    
            ax[0].plot(wav_data_maxL[ifo]['sample_times'], wav_data_maxL[ifo]['strain'], color='red', linestyle='dashdot', label=maxL_label)

            ax[1].plot(wav_data_maxL[ifo]['sample_times'], wav_data_maxL[ifo]['strain'], color='red', linestyle='dashdot', label=maxL_label)
            
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


        ## Plot annotations and settings

        ax[0].grid(True, alpha=0.4)
        ax[1].set_xlabel('Time [s]')
        ax[0].set_ylabel('Strain')
        ax[1].set_ylabel('Strain')
        ax[0].set_xlim(-0.25 + 1e9, 0.25 + 1e9)
        ax[1].legend(loc='upper right')
        ax[0].set_title('Zoomed in')
        if args.taper_domain:
            ax[1].set_xlim(-10.0 + 1e9, 10.0 + 1e9) # for now
            ax[1].set_title('Taper domain')
        else:
            ax[1].set_title('Full domain')

        if args.use_hyperbolic:
            textstr_inj = '\n'.join((
                'Injected:',
                '$m_1 = %.1f \: [M_{\odot}]$' % (P_truth[ifo].m1/SM),
                '$m_2 = %.1f \: [M_{\odot}]$' % (P_truth[ifo].m2/SM),
                '$S_{1,z}=%.2f$' % (P_truth[ifo].s1z),
                '$S_{2,z}=%.2f$' % (P_truth[ifo].s2z),
                '$E_0=%.5f$' % (P_truth[ifo].E0),
                '$p_{\phi}^0=%.5f$' % (P_truth[ifo].p_phi0)))

            ax[0].text(0.0200, 0.98, textstr_inj, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            if args.user_test:
                textstr_test = '\n'.join((
                    'Test waveform:',
                    '$m_1 = %.1f \: [M_{\odot}]$' % (P_test[ifo].m1/SM),
                    '$m_2 = %.1f \: [M_{\odot}]$' % (P_test[ifo].m2/SM),
                    '$S_{1,z}=%.2f$' % (P_test[ifo].s1z),
                    '$S_{2,z}=%.2f$' % (P_test[ifo].s2z),
                    '$E_0=%.5f$' % (P_test[ifo].E0),
                    '$p_{\phi}^0=%.5f$' % (P_test[ifo].p_phi0)))

                ax[0].text(0.2200, 0.98, textstr_test, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            if args.maxL:
                textstr_maxL = '\n'.join((
                    'Max Likelihood:',
                    '$m_1 = %.1f \: [M_{\odot}]$' % (P_maxL[ifo].m1/SM),
                    '$m_2 = %.1f \: [M_{\odot}]$' % (P_maxL[ifo].m2/SM),
                    '$S_{1,z}=%.2f$' % (P_maxL[ifo].s1z),
                    '$S_{2,z}=%.2f$' % (P_maxL[ifo].s2z),
                    '$E_0=%.5f$' % (P_maxL[ifo].E0),
                    '$p_{\phi}^0=%.5f$' % (P_maxL[ifo].p_phi0)))

                ax[0].text(0.1200, 0.98, textstr_maxL, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        else:
            textstr_inj = '\n'.join((
                'Injected:',
                '$m_1 = %.1f \: [M_{\odot}]$' % (P_truth[ifo].m1/SM),
                '$m_2 = %.1f \: [M_{\odot}]$' % (P_truth[ifo].m2/SM),
                '$S_{1,x}=%.2f$' % (P_truth[ifo].s1x),
                '$S_{1,y}=%.2f$' % (P_truth[ifo].s1y),
                '$S_{1,z}=%.2f$' % (P_truth[ifo].s1z),
                '$S_{2,x}=%.2f$' % (P_truth[ifo].s2x),
                '$S_{2,y}=%.2f$' % (P_truth[ifo].s2y),
                '$S_{2,z}=%.2f$' % (P_truth[ifo].s2z),
                '$\chi_p=%.2f$' % (P_truth[ifo].extract_param('chi_p')),
                r'$\sqrt{\langle \chi_p^2 \rangle} =%.2f$' % (P_truth[ifo].extract_param('chi_prms'))))

            ax[0].text(0.0200, 0.98, textstr_inj, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            if args.user_test:
                textstr_test = '\n'.join((
                    'Test waveform:',
                    '$m_1 = %.1f \: [M_{\odot}]$' % (P_test[ifo].m1/SM),
                    '$m_2 = %.1f \: [M_{\odot}]$' % (P_test[ifo].m2/SM),
                    '$S_{1,x}=%.2f$' % (P_test[ifo].s1x),
                    '$S_{1,y}=%.2f$' % (P_test[ifo].s1y),
                    '$S_{1,z}=%.2f$' % (P_test[ifo].s1z),
                    '$S_{2,x}=%.2f$' % (P_test[ifo].s2x),
                    '$S_{2,y}=%.2f$' % (P_test[ifo].s2y),
                    '$S_{2,z}=%.2f$' % (P_test[ifo].s2z),
                    '$\chi_p=%.2f$' % (P_test[ifo].extract_param('chi_p')),
                    r'$\sqrt{\langle \chi_p^2 \rangle} =%.2f$' % (P_truth[ifo].extract_param('chi_prms'))))

                ax[0].text(0.2200, 0.98, textstr_test, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            if args.maxL:
                textstr_maxL = '\n'.join((
                    'Max Likelihood:',
                    '$m_1 = %.1f \: [M_{\odot}]$' % (P_maxL[ifo].m1/SM),
                    '$m_2 = %.1f \: [M_{\odot}]$' % (P_maxL[ifo].m2/SM),
                    '$S_{1,x}=%.2f$' % (P_maxL[ifo].s1x),
                    '$S_{1,y}=%.2f$' % (P_maxL[ifo].s1y),
                    '$S_{1,z}=%.2f$' % (P_maxL[ifo].s1z),
                    '$S_{2,x}=%.2f$' % (P_maxL[ifo].s2x),
                    '$S_{2,y}=%.2f$' % (P_maxL[ifo].s2y),
                    '$S_{2,z}=%.2f$' % (P_maxL[ifo].s2z),
                    '$\chi_p=%.2f$' % (P_maxL[ifo].extract_param('chi_p')),
                    r'$\sqrt{\langle \chi_p^2 \rangle }=%.2f$' % (P_maxL[ifo].extract_param('chi_prms'))))

                ax[0].text(0.1200, 0.98, textstr_maxL, transform=ax[0].transAxes, fontsize=12, horizontalalignment='left', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        lmax_str = f'lmax = {args.lmax}' 

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
            forced_str = f"forced-{ifo}" if args.force_single_IFO else ifo
            if args.maxL and args.user_test:
                if args.use_hyperbolic:
                    savepath = os.path.join(
                        diag_path,
                        f"Injection+maxL+test_mtot-{(P_test[ifo].m1 + P_test[ifo].m2) / SM}_q-{P_test[ifo].m1 / P_test[ifo].m2}_chi1-{P_test[ifo].s1z}_chi2-{P_test[ifo].s2z}_E0-{P_test[ifo].E0}_pphi0-{P_test[ifo].p_phi0}_lmax-{args.lmax}_ILEresults_{forced_str}-frame.png"
                    )
                else:
                    savepath = os.path.join(
                        diag_path,
                        f"Injection+maxL+test_mtot-{(P_test[ifo].m1 + P_test[ifo].m2) / SM}_q-{P_test[ifo].m1 / P_test[ifo].m2}_chi1-{P_test[ifo].s1z}_chi2-{P_test[ifo].s2z}_lmax-{args.lmax}_ILEresults_{forced_str}-frame.png"
                    )
            elif args.maxL and not args.user_test:
                savepath = os.path.join(
                    diag_path,
                    f"Injection+maxL_lmax-{args.lmax}_ILEresults_{forced_str}-frame.png"
                )
            elif args.user_test and not args.maxL:
                if args.use_hyperbolic:
                    savepath = os.path.join(
                        diag_path,
                        f"Injection+test_mtot-{(P_test[ifo].m1 + P_test[ifo].m2) / SM}_q-{P_test[ifo].m1 / P_test[ifo].m2}_chi1-{P_test[ifo].s1z}_chi2-{P_test[ifo].s2z}_E0-{P_test[ifo].E0}_pphi0-{P_test[ifo].p_phi0}_lmax-{args.lmax}_ILEresults_{forced_str}-frame.png"
                    )
                else:
                    savepath = os.path.join(
                        diag_path,
                        f"Injection+test_mtot-{(P_test[ifo].m1 + P_test[ifo].m2) / SM}_q-{P_test[ifo].m1 / P_test[ifo].m2}_chi1-{P_test[ifo].s1z}_chi2-{P_test[ifo].s2z}_lmax-{args.lmax}_ILEresults_{forced_str}-frame.png"
                    )
            else:
                savepath = os.path.join(
                    diag_path, f"Injection_lmax-{args.lmax}_ILEresults_{forced_str}-frame.png"
                )
                
        else:
            forced_str = f"forced-{ifo}" if args.force_single_IFO else ifo
            if args.maxL and args.user_test:
                if args.use_hyperbolic:
                    savepath = os.path.join(
                        diag_path,
                        f"Injection+maxL+test_mtot-{(P_test[ifo].m1 + P_test[ifo].m2) / SM}_q-{P_test[ifo].m1 / P_test[ifo].m2}_chi1-{P_test[ifo].s1z}_chi2-{P_test[ifo].s2z}_E0-{P_test[ifo].E0}_pphi0-{P_test[ifo].p_phi0}_lmax-{args.lmax}_NoILE_{forced_str}-frame.png"
                    )
                else:
                    savepath = os.path.join(
                        diag_path,
                        f"Injection+maxL+test_mtot-{(P_test[ifo].m1 + P_test[ifo].m2) / SM}_q-{P_test[ifo].m1 / P_test[ifo].m2}_chi1-{P_test[ifo].s1z}_chi2-{P_test[ifo].s2z}_lmax-{args.lmax}_NoILE_{forced_str}-frame.png"
                    )
            elif args.maxL and not args.user_test:
                savepath = os.path.join(
                    diag_path,
                    f"Injection+maxL_lmax-{args.lmax}_NoILE_{forced_str}-frame.png"
                )
            elif args.user_test and not args.maxL:
                if args.use_hyperbolic:
                    savepath = os.path.join(
                        diag_path,
                        f"Injection+test_mtot-{(P_test[ifo].m1 + P_test[ifo].m2) / SM}_q-{P_test[ifo].m1 / P_test[ifo].m2}_chi1-{P_test[ifo].s1z}_chi2-{P_test[ifo].s2z}_E0-{P_test[ifo].E0}_pphi0-{P_test[ifo].p_phi0}_lmax-{args.lmax}_NoILE_{forced_str}-frame.png"
                    )
                else:
                    savepath = os.path.join(
                        diag_path,
                        f"Injection+test_mtot-{(P_test[ifo].m1 + P_test[ifo].m2) / SM}_q-{P_test[ifo].m1 / P_test[ifo].m2}_chi1-{P_test[ifo].s1z}_chi2-{P_test[ifo].s2z}_lmax-{args.lmax}_NoILE_{forced_str}-frame.png"
                    )
            else:
                savepath = os.path.join(
                    diag_path, f"Injection_lmax-{args.lmax}_NoILE_{forced_str}-frame.png"
                )
            


        plt.savefig(savepath, bbox_inches='tight')

else:
    print('No plots enabled, program is complete.')