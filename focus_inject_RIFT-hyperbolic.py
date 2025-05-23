#! /usr/bin/env python
#
# GOAL: single-button injection setup from ini file
#
#
# Note: this one is only for the hyperbolic project.


################################################################
#### Preamble ####
################################################################

import numpy as np
import argparse
import os
import sys
sys.path.append(os.getcwd())
import shutil
import ast
import glob
import re
import configparser

from RIFT.misc.dag_utils import mkdir
import RIFT.lalsimutils as lalsimutils
import lal
import lalsimulation as lalsim

from gwpy.timeseries import TimeSeries
from matplotlib import pyplot as plt

# Backward compatibility
from RIFT.misc.dag_utils import which
lalapps_path2cache = which('lal_path2cache')
if lalapps_path2cache == None:
    lalapps_path2cache =  which('lalapps_path2cache')
    
################################################################
#### Functions & Definitions ####
################################################################


def unsafe_config_get(config,args,verbose=False):
    if verbose:
        print(" Retrieving ", args, end=' ') 
        print(" Found ",eval(config.get(*args)))
    return eval( config.get(*args))

def add_frames(channel,input_frame, noise_frame,combined_frame):
    # note - this needs to be updated to work with the current install regardless of user
    exe = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pp/add_frames.py'))
    if not(os.path.dirname(exe) in os.environ['PATH'].split(os.pathsep)):
        print('add_frames.py not found, adding to PATH...')
        os.environ['PATH'] += os.pathsep + os.path.dirname(exe)
    cmd = exe + " " + channel + " " + input_frame + " " + noise_frame + " " + combined_frame 
    print(cmd)
    os.system(cmd)

lsu_MSUN=lal.MSUN_SI
lsu_PC=lal.PC_SI        

################################################################
#### Options Parse ####
################################################################
        
parser = argparse.ArgumentParser()
parser.add_argument("--use-ini", default=None, type=str, required=True, help="REQUIRED. Pass ini file for parsing. Intended to reproduce lalinference_pipe functionality. Overrides most other arguments.")
parser.add_argument("--use-osg",action='store_true',help="Attempt OSG operation. Command-line level option, not at ini level to make portable results")
parser.add_argument("--add-extrinsic",action='store_true',help="Add extrinsic posterior.  Corresponds to --add-extrinsic --add-extrinsic-time-resampling --batch-extrinsic for pipeline")
parser.add_argument("--force-cpu",action='store_true',help="Forces avoidance of GPUs")
parser.add_argument("--bypass-frames",action='store_true',help="Skip making mdc and frame files, use with caution")
parser.add_argument("--just-frames",action='store_true',help="Stop after making frame files, use with caution")
parser.add_argument("--hlmoft-frames",action='store_true',help="If enabled, builds h(t) frames from hlm(t)")
parser.add_argument("--use-osdf", action='store_true', help='Uses OSDF file transfer instead of CVMFS')
parser.add_argument("--force-frame-srate", action='store_true', help='Uses srate from the config file for frame generation')
opts =  parser.parse_args()

config = configparser.ConfigParser(allow_no_value=True) #SafeConfigParser deprecated from py3.2

ini_path = os.path.abspath(opts.use_ini)

# read in config
config.read(ini_path)

bypass_frames=opts.bypass_frames 
just_frames=opts.just_frames

dist_marg_arg = config.get('rift-pseudo-pipe','internal-marginalize-distance') 
dist_marg = True if dist_marg_arg.lower() == 'true' else False

################################################################
#### Initialize ####
################################################################

# Create, go to working directory
working_dir = config.get('init','working_directory')
print(" Working dir ", working_dir)
mkdir(working_dir)
os.chdir(working_dir)
working_dir_full = os.getcwd()

# load IFO list from config
ifos = unsafe_config_get(config,['analysis','ifos'])

# segment length
seglen = int(config.get('engine','seglen'))

# PSDs
PSD_files = unsafe_config_get(config,["make_psd",'fiducial_psds'])

# frequency bounds - used for PSD settings
PSD_freqs = {}
for ifo in ifos:
    f_low = unsafe_config_get(config, ['lalinference', 'flow'])[ifo]   
    f_high = unsafe_config_get(config, ['lalinference', 'fhigh'])[ifo]
    
    PSD_freqs[ifo] = (f_low, f_high)
    

################################################################
#### Load Injection Parameters from config into lalsimutils ####
################################################################



P_list = []
P = lalsimutils.ChooseWaveformParams()

## Waveform Model ##
approx_str = config.get('engine','approx')
P.approx=lalsim.GetApproximantFromString(approx_str)

## Intrinsic Parameters ##
P.m1 = float(config.get('injection-parameters','m1'))*lsu_MSUN
P.m2 = float(config.get('injection-parameters','m2'))*lsu_MSUN
P.s1x = float(config.get('injection-parameters','s1x'))
P.s1y = float(config.get('injection-parameters','s1y'))
P.s1z = float(config.get('injection-parameters','s1z'))
P.s2x = float(config.get('injection-parameters','s2x'))
P.s2y = float(config.get('injection-parameters','s2y'))
P.s2z = float(config.get('injection-parameters','s2z'))
P.E0 = float(config.get('injection-parameters','E0'))
P.p_phi0 = float(config.get('injection-parameters','p_phi0'))
## Extrinsic Parameters ##
P.dist = float(config.get('injection-parameters','dist'))*1e6*lsu_PC
fiducial_event_time = float(config.get('injection-parameters','event_time')) # for frame addition
P.tref = fiducial_event_time
P.theta = float(config.get('injection-parameters','dec'))
P.phi = float(config.get('injection-parameters','ra'))
P.incl = float(config.get('injection-parameters','incl'))
P.psi = float(config.get('injection-parameters','psi'))
P.phiref = float(config.get('injection-parameters','phase'))
# Other waveform settings
P.fmin = float(config.get('injection-parameters','fmin'))

hypclass = P.extract_param('hypclass')
print(f'Injection type is {hypclass}')

lmax = int(config.get('rift-pseudo-pipe','l-max'))
   
P_list.append(P)

# create mdc.xml.gz - contains all injection params
if bypass_frames:
    print("Skipping mdc creation")
    pass
else:
    mdc_name = 'mdc'        
    lalsimutils.ChooseWaveformParams_array_to_xml(P_list,f"{mdc_name}")

    
    
################################################################    
#### Create frame files and cache file ####
################################################################

if bypass_frames:
    print("Skipping frame creation")
    indx = 0
    pass
else:
    # create clean signal frames #
    print(" === Writing signal files === ")

    mkdir('signal_frames')
    # must be compatible with duration of the noise frames
    t_start = int(fiducial_event_time)-150
    #t_start = int(fiducial_event_time)-75
    t_stop = int(fiducial_event_time)+150
    #t_stop = int(fiducial_event_time)+75

    indx = 0 # need an event number, meaningless otherwise
    target_subdir = 'signal_frames/event_{}'.format(indx) # this is where the signal frames will go
    mkdir(working_dir_full+"/"+target_subdir)
    os.chdir(working_dir_full+"/"+target_subdir)
    # here we're in the signal_frames/event_0 directory.
    # Loop over instruments, write frame files and cache
    for ifo in ifos:
        cmd = "util_LALWriteFrame.py --inj " + working_dir_full+"/{}.xml.gz --event {} --start {}  --stop {}  --instrument {} --approx {} --seglen {} --l-max {} --hyperbolic  ".format(mdc_name, 0, t_start,t_stop,ifo, approx_str, seglen, lmax) # note that event is always zero here
        #cmd = "util_LALWriteFrame.py --inj " + working_dir_full+"/{}.xml.gz --event {} --instrument {} --approx {} --seglen {} --l-max {} --hyperbolic  ".format(mdc_name, 0,ifo, approx_str, seglen, lmax) # note that event is always zero here
        if opts.hlmoft_frames:
            cmd += ' --gen-hlmoft '
            
        if opts.force_frame_srate:
            fs = int(float(config.get('engine','srate')))
            cmd += f' --srate {fs} '
            
            
        print(cmd)
        os.system(cmd)
        
    # Here we plot the signal frames    
    gwf_files = [os.path.abspath(file) for file in os.listdir(os.getcwd()) if file.endswith('.gwf')]
    
    for index, path in enumerate(gwf_files):
        file = os.path.basename(path)
        # assign correct channel name
        if file.startswith('H'):
            channel = 'H1:FAKE-STRAIN'
        elif file.startswith('L'):
            channel = 'L1:FAKE-STRAIN'
        elif file.startswith('V'):
            channel = 'V1:FAKE-STRAIN'
            
        data = TimeSeries.read(path, channel)
        strain = np.array(data.value)
        sample_times = np.array(data.times)
        
        # plot domain 
        max_amp = np.argmax(strain)
        #seglen = int(config.get('engine','seglen'))
        #domain_start = seglen - 2 + 1
        if hypclass == 'scatter':            
            plot_domain = [sample_times[max_amp] - 0.25, sample_times[max_amp] + 0.25]
            
        elif hypclass == 'plunge':
            plot_domain = [sample_times[max_amp] - 0.5, sample_times[max_amp] + 0.5]
            
        elif hypclass == 'zoomwhirl':
            plot_domain = [sample_times[max_amp] - 8.0, sample_times[max_amp] + 1.0]
        
        plt.plot(sample_times, data, label=f'{channel}')
        plt.legend(loc='best')
        plt.savefig(f'{channel}_plot_full.png', bbox_inches='tight')        
        plt.xlim(plot_domain)
        plt.savefig(f'{channel}_plot_{hypclass}-domain.png', bbox_inches='tight')
        plt.close()
             
    
    # write frame paths to cache
    os.chdir(working_dir_full)
    target_subdir='signal_frames/event_{}'.format(indx)
    os.chdir(working_dir_full+"/"+target_subdir)
    cmd = "/bin/find .  -name '*.gwf' | {} > signals.cache".format(lalapps_path2cache)
    os.system(cmd)
    
    

# Calculate SNRs for zero-noise frames
target_subdir='signal_frames/event_{}'.format(indx)
target_subdir_full = working_dir_full+"/"+target_subdir
os.chdir(target_subdir_full)
if os.path.exists(target_subdir_full + '/snr-report.txt'):
    print('SNR report already exists, skipping')
    pass
else:
    cmd = "util_FrameZeroNoiseSNR.py --cache signals.cache --plot-sanity --verbose"
    fmin_snr, fmax_snr = PSD_freqs["H1"]
    cmd += f" --fmin-snr {fmin_snr} --fmax-snr {fmax_snr} "
    for ifo in ifos:
        cmd += f" --psd-file {ifo}={PSD_files[ifo]} "
        
            
        print(cmd)
        os.system(cmd)
with open('snr-report.txt', 'r') as file:
    snr_dict = ast.literal_eval(file.read())
file.close()
snr_guess = snr_dict['Network'] # passed to pseudo_pipe as --force-hint-snr
    
if opts.just_frames:
    print('Exiting after frame creation!')
    sys.exit(0)

################################################################
#### RIFT Workflow Generation ####
################################################################ 


# check for distance marginalization lookup table - primarily for testing
if bypass_frames:
    os.chdir(working_dir_full)
    if os.path.exists(working_dir_full+'/'+'distance_marginalization_lookup.npz'):
        print("Using existing distance marginalization table")
        dist_marg_exists = True
        dist_marg_path = working_dir_full + "/distance_marginalization_lookup.npz"
    else:
        if dist_marg:
            print("Distance marginalization table not found, it will be made during the workflow")
        dist_marg_exists = False
else:
    dist_marg_exists = False
    

os.chdir(working_dir_full)

if not(bypass_frames):
    # write coinc file
    cmd = "util_SimInspiralToCoinc.py --sim-xml {}.xml.gz --event {} ".format(mdc_name, 0) # Note that the event getting passed is always zero here, but not the name
    for ifo in ifos:
        cmd += "  --ifo {} ".format(ifo)
        
    os.system(cmd)
    coinc_name = 'coinc.xml'
else:
    # still need the coinc name
    coinc_name = 'coinc.xml'
    

# Designate rundir 
rundir = config.get('init','rundir')
    
#point to signals.cache
target_file = 'signal_frames/event_{}/signals.cache'.format(indx)

# RIFT analysis setup
cmd = 'util_RIFT_pseudo_pipe.py --use-coinc `pwd`/{} --use-ini {} --use-rundir `pwd`/{} --fake-data-cache `pwd`/{} --force-hint-snr {} --assume-hyperbolic '.format(coinc_name, ini_path, rundir, target_file, snr_guess)
if opts.add_extrinsic:
    cmd += ' --add-extrinsic --add-extrinsic-time-resampling --batch-extrinsic '
if dist_marg_exists:
    cmd += ' --internal-marginalize-distance-file {} '.format(dist_marg_path)
if opts.force_cpu:
    if opts.use_osg:
        print("Don't force CPUs on the OSG, exiting.")
        sys.exit(1)
    else:
        print('Avoiding GPUs')
        cmd += ' --ile-no-gpu '
if opts.use_osg:
    print('Configuring for OSG operation')
    cmd += ' --use-osg '
    cmd += ' --use-osg-file-transfer '
    cmd += ' --ile-retries 10 '
    # note: --use-osg-cip not used by default - can set this in the ini file
    
os.system(cmd)   
          
################################################################
#### Post-Workflow Cleanup ####
################################################################

# copy over PSDs
psd_dict = unsafe_config_get(config,["make_psd",'fiducial_psds'])
for psd in psd_dict:
    psd_path = psd_dict[psd]
    shutil.copy2(psd_path, rundir)

# copy config file into rundir/reproducibility
repro_path = working_dir_full+"/"+rundir+"/"+"reproducibility"
shutil.copy2(ini_path, repro_path)

if opts.use_osg:
    # make frames_dir and copy frame files for transfer
    os.chdir(working_dir_full+"/"+rundir)
    mkdir('frames_dir')
    os.chdir(working_dir_full+"/combined_frames/event_{}".format(indx))
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.gwf'):
            pathname = os.path.join(os.getcwd(), filename)
            if os.path.isfile(pathname):
                shutil.copy2(pathname, working_dir_full+"/"+rundir+"/frames_dir")
                
    
# for ease of testing, copy the distance marginalization table to the top-level dir
if dist_marg:
    if os.path.exists(working_dir_full+'/'+'distance_marginalization_lookup.npz'):
        pass
    else:
        print('Copying distance marginalization')
        os.chdir(working_dir_full+"/"+rundir)
        shutil.copy2('distance_marginalization_lookup.npz', working_dir_full)
    

# edit dag file + change name of dag
os.chdir(working_dir_full+"/"+rundir)

os.rename('marginalize_intrinsic_parameters_BasicIterationWorkflow.dag', working_dir + '.dag')

## Update ILE requirements to avoid bad hosts ##

excluded_hosts = ["(TARGET.Machine =!= 'deepclean.ldas.cit')"]

#host_file = '/home/richard.oshaughnessy/igwn_feedback/rift_avoid_hosts.txt'

#with open(host_file, 'r') as file:
#    for line in file:
#        hostname = line.strip()
#        excluded_hosts.append(f"(TARGET.Machine =!= '{hostname}')")

avoid_string = "&&".join(excluded_hosts)

# List of file paths to modify
file_paths = ["ILE.sub","ILE_puff.sub","iteration_4_cip/ILE.sub","iteration_4_cip/ILE_puff.sub"]
      
if config.getboolean('rift-pseudo-pipe', 'internal-use-aligned-phase-coordinates'):  
    file_paths = ["ILE.sub","ILE_puff.sub","iteration_2_cip/ILE.sub","iteration_2_cip/ILE_puff.sub"]
else:
    file_paths = ["ILE.sub","ILE_puff.sub","iteration_3_cip/ILE.sub","iteration_3_cip/ILE_puff.sub"]
    
if opts.add_extrinsic:
    file_paths.append("ILE_extr.sub")

for file_path in file_paths:
    # Construct the full file path
    full_path = os.path.join(os.getcwd(), file_path)

    # Read the contents of the file
    with open(full_path, "r") as file:
        lines = file.readlines()
        
    # Find the index of the line containing the 'requirements' command
    requirements_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("requirements"):
            requirements_index = i
            break
            
    # Find the index of the line containing the 'queue' command
    queue_index = len(lines) - 1
    for i, line in enumerate(reversed(lines)):
        if line.strip().startswith("queue"):
            queue_index = len(lines) - 1 - i
            break

    # Find the index of the line containing 'request_memory'
    memory_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("request_memory"):
            memory_index = i
            break
    
    # Modify lines
    if requirements_index is not None:
        #requirements_line = lines[requirements_index]
        if opts.use_osg:
            requirements_line = "requirements = (HAS_SINGULARITY=?=TRUE)&&(IS_GLIDEIN)&&(HAS_CVMFS_singularity_opensciencegrid_org=?=TRUE)" + "&&" + avoid_string + "\n"
            lines[requirements_index] = requirements_line
            # Insert the 'require_gpus' line before the 'queue' command
            #lines.insert(queue_index, "require_gpus = Capability >= 3.5\n")
            lines.insert(queue_index, "gpus_minimum_capability = 3.5\n")            
            lines.insert(queue_index, "gpus_minimum_memory = 6GB\n")
            lines.insert(queue_index, 'MY.UNDESIRED_Sites = "SDSC-PRP"\n')
        else:
            requirements_line = "requirements = " + avoid_string + "\n"
            lines[requirements_index] = requirements_line

    # Modify the 'request_memory' line 
    if memory_index is not None:
        lines[memory_index] = "request_memory = 16384M\n"
            
    # Write the modified contents back to the file
    with open(full_path, "w") as file:
        file.writelines(lines)

#
#
## special configuration for hyperbolic analysis in the IGWN pool
if opts.use_osg:
    
    print('Modifying submit files for hyperbolics in the IGWN pool')
    
    # need to edit all ILE and CIP submit files
    if config.getboolean('rift-pseudo-pipe', 'internal-use-aligned-phase-coordinates'):
        ILE_file_paths = ["ILE.sub","ILE_puff.sub","iteration_2_cip/ILE.sub","iteration_2_cip/ILE_puff.sub"]
        CIP_file_paths = ["CIP.sub", "CIP_worker.sub", "CIP_0.sub", "CIP_worker0.sub", "CIP_1.sub", "CIP_worker1.sub", "CIP_2.sub", "CIP_worker2.sub", "iteration_2_cip/CIP.sub", "iteration_2_cip/CIP_worker.sub"]
        PUFF_file_paths = ["PUFF.sub", "iteration_2_cip/PUFF.sub"]
    else:
        ILE_file_paths = ["ILE.sub","ILE_puff.sub","iteration_3_cip/ILE.sub","iteration_3_cip/ILE_puff.sub"]
        CIP_file_paths = ["CIP.sub", "CIP_worker.sub", "CIP_0.sub", "CIP_worker0.sub", "CIP_1.sub", "CIP_worker1.sub", "CIP_2.sub", "CIP_worker2.sub", "iteration_3_cip/CIP.sub", "iteration_3_cip/CIP_worker.sub"]
        PUFF_file_paths = ["PUFF.sub", "iteration_3_cip/PUFF.sub"]
    

    if opts.add_extrinsic:
        ILE_file_paths.append("ILE_extr.sub")
        
        
    total_file_paths = ILE_file_paths + CIP_file_paths
    
    
    def modify_submit_file(input_file):
        
        #current_singularity_image = 'rift_o4b_jl-chadhenshaw-teobresums_eccentric-2024-05-14_12-02-56.sif'
        #sif_path = 'osdf:///igwn/cit/staging/james.clark/' + current_singularity_image
        
        if opts.use_osdf:
            current_singularity_image = 'rift_o4b_jl-2024-09-25_12-39-46.sif'
            sif_path = 'osdf:///igwn/cit/staging/chad.henshaw/' + current_singularity_image
        else:    
            current_singularity_image = '/cvmfs/singularity.opensciencegrid.org/james-clark/research-projects-rit/containers-rift_o4b_jl-chadhenshaw-teobresums_eccentric:latest'
        
        with open(input_file, 'r') as file:
            lines = file.readlines()
            
        if opts.use_osdf:
            lines_to_remove = ['getenv', '+SingularityBindCVMFS', '+SingularityImage']
            lines_to_add = [f'MY.SingularityImage = "{current_singularity_image}"', 'use_oauth_services = scitokens']
            modifications = {'request_disk': 'request_disk = 3G', 'requirements': 'requirements = (HAS_SINGULARITY=?=TRUE)&&(IS_GLIDEIN=?=TRUE)&&(TARGET.Machine =!= "deepclean.ldas.cit")', 'transfer_input_files': None}
        
        else:
            lines_to_remove = ['getenv']
            lines_to_add = []
            modifications = {'request_disk': 'request_disk = 50M', 'requirements': 'requirements = (HAS_SINGULARITY=?=TRUE)&&(IS_GLIDEIN=?=TRUE)&&(TARGET.Machine =!= "deepclean.ldas.cit")&&(HAS_CVMFS_singularity_opensciencegrid_org=?=TRUE)'}

        modified_lines = []
        queue_line = None

        for line in lines:
            stripped_line = line.strip()

            # remove lines
            if any(stripped_line.startswith(keyword) for keyword in lines_to_remove):
                continue

            # modify lines
            modified = False
            for key, new_value in modifications.items():
                if stripped_line.startswith(key):
                    if key == 'transfer_input_files':
                        if opts.use_osdf:
                            line = stripped_line + ',' + sif_path + '\n'
                        else:
                            print('skipping this for cvfms')
                    else:
                        line = new_value + '\n'
                    modified_lines.append(line)
                    modified = True
                    break
                    
            if stripped_line.startswith('queue'):
                queue_line = line
            elif not modified:
                modified_lines.append(line)
                

        # add new lines
        modified_lines.extend([line + '\n' for line in lines_to_add])
        
        if queue_line:
            modified_lines.append(queue_line)

        with open(input_file, 'w') as file:
            file.writelines(modified_lines)
            
        return
    
    def modify_CIP_coords(input_file, mtot_range):
        with open(input_file, 'r') as file:
            lines = file.readlines()

        # Format the new mtot_range string
        mtot_range_str = f"[{mtot_range[0]}, {mtot_range[1]}]"

        for i, line in enumerate(lines):
            if line.startswith("arguments ="):
                # Replace --parameter mc with --parameter mtot
                line = line.replace('--parameter mc', '--parameter mtot')
                # Use regular expression to find and replace the --mc-range argument
                line = re.sub(r'--mc-range\s*\'\[.*?\]\'', f"--mtot-range '{mtot_range_str}'", line)
                lines[i] = line

        with open(input_file, 'w') as file:
            file.writelines(lines)
    
        return
    
    def modify_puff_coords(input_file):
        with open(input_file, 'r') as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            if line.startswith("arguments ="):
                # Replace --parameter mc with --parameter mtot
                line = line.replace('--parameter mc', '--parameter mtot')
                
                lines[i] = line
        with open(input_file, 'w') as file:
            file.writelines(lines)
    
        return
    
    for submit_file in total_file_paths:
        modify_submit_file(submit_file)
