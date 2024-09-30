#!/usr/bin/env python3



import argparse
import os
import glob
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for finding grid points missing from RIFT composite files.')
    parser.add_argument('--analysis-dir', required=True, type=str, help='Absolute path to the analysis directory.')
    return parser.parse_args()

# Parse arguments
args = parse_arguments()


rundir = args.analysis_dir
basedir = os.path.dirname(rundir)

# Create diagnostic directory
diag_path = os.path.join(rundir, 'diagnostic')
if not(os.path.exists(diag_path)):
    os.mkdir(diag_path)
    
# combined ile results path 
combined_file_path = os.path.join(diag_path, 'combined_ile_results.dat')

# Open the combined file in write mode
with open(combined_file_path, 'w') as combined_file:
    # Loop through the iteration_x_ile directories
    for iteration_dir in glob.glob(os.path.join(rundir, 'iteration_*_ile')):
        # Find all .dat files that match the pattern CME*.dat
        for dat_file in glob.glob(os.path.join(iteration_dir, 'CME*.dat')):
            # Open each .dat file and append its contents to the combined file
            with open(dat_file, 'r') as infile:
                combined_file.write(infile.read())

print(f"All .dat files combined into {combined_file_path}")

# Now run the external script util_CleanILE_test.py on combined_ile_results.dat
output_file_path = os.path.join(diag_path, 'test.composite')
try:
    subprocess.run(
        ['python3', '/home/chad.henshaw/RIFT_henshaw/RIFT_tools/util_CleanILE_test.py', 
         '--analysis-dir', rundir, combined_file_path], 
        stdout=open(output_file_path, 'w'), 
        check=True
    )
    print(f"util_CleanILE_test.py ran successfully, output saved to {output_file_path}")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while running util_CleanILE_test.py: {e}")