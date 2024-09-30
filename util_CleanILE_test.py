#!/usr/bin/env python3
# Test version of util_CleanILE.py to identify missing grid points
#
#  Reads FILE (not stdin). Consolidates ILE entries for the same physical system.
#  Compare to: util_MassGriCoalesce.py


import sys
import os
import RIFT.misc.xmlutils as xmlutils
#from optparse import OptionParser
from ligo.lw import lsctables, table, utils

import numpy as np
import RIFT.misc.weight_simulations as weight_simulations

import fileinput
#import StringIO

data_at_intrinsic = {}

my_digits=5  # safety for high-SNR BNS

tides_on = False
distance_on = False  
col_intrinsic = 9

import argparse
parser = argparse.ArgumentParser(usage="util_CleanILE.py fname1.dat fname2.dat ... ")
parser.add_argument("fname",action='append',nargs='+')
parser.add_argument("--eccentricity", action="store_true")
#Askold: adding specification for tabular eos file
parser.add_argument("--tabular-eos-file", action="store_true") 
parser.add_argument('--analysis-dir', required=True, type=str, help='Absolute path to the analysis directory.')
opts = parser.parse_args()

# exclusion log file
log_file = open(os.path.join(opts.analysis_dir, 'diagnostic/SigmaOverL_exclusions.txt'), 'w')

# initialize grid point counts
total_input_lines = 0
sigma_over_l_exclusions = 0
duplicate_consolidations = 0

#print opts.fname
from pathlib import Path
for fname in opts.fname[0]: #sys.argv[1:]:
    fname  = Path(fname).resolve()
    if not( os.path.exists(fname)):  # skip symbolic links that don't resolve : important for .composite files
        print(f"Skipping {fname}: file does not exist", file=sys.stderr)
        continue
    if os.stat(fname).st_size==0:  # skip files of zero length
        print(f"Skipping {fname}: file is empty", file=sys.stderr)
        continue
    sys.stderr.write(str(fname)+"\n")
#    data = np.loadtxt(fname)  # this will FAIL if we have a heterogeneous data source!  BE CAREFUL
    data = np.genfromtxt(fname,invalid_raise=False)  #  Protect against inhomogeneous data
    if len(data.shape) ==1:
        data = np.array([data]) # force proper treatment for single-line file
    for line in data:
      total_input_lines += 1  # Track total number of input lines  
      try:
        line = np.around(line, decimals=my_digits)
        lambda1=lambda2=0
        eos_index = 0
        if opts.eccentricity:
            indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z,ecc, lnL, sigmaOverL, ntot, neff = line
            col_intrinsic = 10
        elif len(line) == 13 and (not tides_on) and (not distance_on):  # strip lines with the wrong length
            indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z,lnL, sigmaOverL, ntot, neff = line
        elif  len(line) == 14:
            distance_on=True
            col_intrinsic=10
            indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z,dist, lnL, sigmaOverL, ntot, neff = line
        elif len(line)==15:
            tides_on  = True
            col_intrinsic =11
            indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z, lambda1,lambda2,lnL, sigmaOverL, ntot, neff = line

        #Askold: adding the option for tabular eos file
        elif opts.tabular_eos_file and len(line) == 16: #checking if the tabular eos file is defined in the parser and if the line actually has all the columns
            #no eccentricity assumed here, since export_eos_index option doesn't output eccentricity, also it doesn't apply to neutron stars
            col_intrinsic = 12 #I assume eos_index to be intrinsic parameter
            indx, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, lambda1, lambda2, eos_index, lnL, sigmaOverL, ntot, neff = line 

        if sigmaOverL>0.9:
            #print(f"Excluding line due to high sigmaOverL: {line}")
            sigma_over_l_exclusions += 1
            #print(line, file=sys.stderr)
            formatted_line = ' '.join(map(str, line))
            log_file.write(f'{formatted_line}\n')
            continue    # do not allow poorly-resolved cases (e.g., dominated by one point). These are often useless
        if tuple(line[1:col_intrinsic]) in data_at_intrinsic:
            duplicate_consolidations += 1
            #print(f"Consolidating duplicate entry: {tuple(line[1:col_intrinsic])} (Line: {line})", file=sys.stderr)
#            print " repeated occurrence ", line[1:9]
            data_at_intrinsic[tuple(line[1:col_intrinsic])].append(line[col_intrinsic:])
        else:
#            print " new key ", line[1:9]
            data_at_intrinsic[tuple(line[1:col_intrinsic])] = [line[col_intrinsic:]]
      except Exception as e:
          print(f"Error processing line: {line}. Exception: {e}")
          continue
        

for key in data_at_intrinsic:
    lnL, sigmaOverL, ntot,neff =   np.transpose(data_at_intrinsic[key])
    sigmaOverL = np.maximum(sigmaOverL, 1e-7*np.ones(len(lnL)))   # prevent accidental underflow during debugging/using synthetic data with no error
    lnLmax = np.max(lnL)
    sigma = sigmaOverL*np.exp(lnL-lnLmax)  # remove overall Lmax factor, which factors out from the weights constructed from \sigma
    wts = weight_simulations.AverageSimulationWeights(None, None,sigma)   
    lnLmeanMinusLmax = np.log(np.sum(np.exp(lnL - lnLmax)*wts))
    sigmaNetOverL = (np.sqrt(1./np.sum(1./sigma/sigma)))/np.exp(lnLmeanMinusLmax)


    if opts.eccentricity:
        print(-1,  key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], key[8], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
    elif tides_on:
        print(-1,  key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], key[8],key[9], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
    elif distance_on:
        print(-1,  key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], key[8], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
    
    #Askold: new option for tabular eos file
    elif opts.tabular_eos_file: #written similarly to the previous ones
        print(-1, key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], key[8],key[9], key[10],  lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
    
    else:
        print(-1,  key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)

        
        
log_file.close()

# Save summary to missing_points_summary.txt
summary_file_path = os.path.join(opts.analysis_dir, 'diagnostic/missing_points_summary.txt')
with open(summary_file_path, 'w') as summary_file:
    summary_file.write(f"Total input lines: {total_input_lines}\n")
    summary_file.write(f"Excluded due to SigmaOverL > 0.9: {sigma_over_l_exclusions}\n")
    summary_file.write(f"Consolidated as duplicates: {duplicate_consolidations}\n")
    summary_file.write(f"Output lines: {len(data_at_intrinsic)}\n")