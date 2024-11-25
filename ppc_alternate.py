#!/usr/bin/env python3

# Alternate version of plot_posterior_corner.py
#

import RIFT.lalsimutils as lalsimutils
import RIFT.misc.samples_utils as samples_utils
from RIFT.misc.samples_utils import add_field, extract_combination_from_LI, standard_expand_samples
import lal
import numpy as np
import argparse
import numpy.lib.recfunctions as rfn

eos_param_names = ['logp1', 'gamma1','gamma2', 'gamma3', 'R1_km', 'R2_km']


try:
    import matplotlib
    print(" Matplotlib backend ", matplotlib.get_backend())
    if matplotlib.get_backend() == 'agg':
        fig_extension = '.png'
        bNoInteractivePlots=True
    else:
        matplotlib.use('agg')
        fig_extension = '.png'
        bNoInteractivePlots =True
    from matplotlib import pyplot as plt
    bNoPlots=False
except:
    print(" Error setting backend")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
import corner

import RIFT.misc.our_corner as our_corner
try:
    import RIFT.misc.bounded_kde as bounded_kde
except:
    print(" -No 1d kdes- ")


print(" WARNINGS : BoundedKDE class can oversmooth.  Need to edit options for using this class! ")

def render_coord(x,logscale=False):
    if x in lalsimutils.tex_dictionary.keys():
        mystr= lalsimutils.tex_dictionary[x]
        if logscale:
            mystr=mystr.lstrip('$')
            mystr = r"$\log_{10}"+mystr
            return mystr
        else:
            return mystr
    if 'product(' in x:
        a=x.replace(' ', '') # drop spaces
        a = a[:len(a)-1] # drop last
        a = a[8:]
        terms = a.split(',')
        exprs =list(map(render_coord, terms))
        exprs = list(map( lambda x: x.replace('$', ''), exprs))
        my_label = ' '.join(exprs)
        return '$'+my_label+'$'
    else:
        if logscale:
            return "log10 "+str(x)
        return x

def render_coordinates(coord_names,logparams=[]):
    print("log params ",logparams)
    return list(map(lambda x: render_coord(x,logscale=(x in logparams)), coord_names))




remap_ILE_2_LI = samples_utils.remap_ILE_2_LI
remap_LI_to_ILE = samples_utils.remap_LI_to_ILE

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--posterior-file",action='append',help="filename of *.dat file [standard LI output]")
parser.add_argument("--truth-file",type=str, help="file containing the true parameters")
parser.add_argument("--truth-file-manual",type=str, help="file containing the true parameters. Use labelled columns")
parser.add_argument("--posterior-distance-factor",action='append',help="Sequence of factors used to correct the distances")
parser.add_argument("--truth-event",type=int, default=0,help="file containing the true parameters")
parser.add_argument("--composite-file",action='append',help="filename of *.dat file [standard ILE intermediate]")
parser.add_argument("--composite-file-has-labels",action='store_true',help="Assume header for composite file")
parser.add_argument("--use-all-composite-but-grayscale",action='store_true',help="Composite")
parser.add_argument("--flag-tides-in-composite",action='store_true',help='Required, if you want to parse files with tidal parameters')
parser.add_argument("--flag-eos-index-in-composite",action='store_true',help='Required, if you want to parse files with EOS index in composite (and tides)')
parser.add_argument("--posterior-label",action='append',help="label for posterior file")
parser.add_argument("--posterior-color",action='append',help="color and linestyle for posterior. PREPENDED onto default list, so defaults exist")
parser.add_argument("--posterior-linestyle",action='append',help="color and linestyle for posterior. PREPENDED onto default list, so defaults exist")
parser.add_argument("--parameter", action='append',help="parameter name (ILE). Note source-frame masses are only natively supported for LI")
parser.add_argument("--parameter-log-scale",action='append',help="Put this parameter in log scale")
parser.add_argument("--change-parameter-label", action='append',help="format name=string. Will be wrapped in $...$")
parser.add_argument("--use-legend",action='store_true')
parser.add_argument("--use-title",default=None,type=str)
parser.add_argument("--use-smooth-1d",action='store_true')
parser.add_argument("--plot-1d-extra",action='store_true')
parser.add_argument("--pdf",action='store_true',help="Export PDF plots")
#option deprecated by bind-param and param-bound
#parser.add_argument("--mc-range",default=None,help='List for mc range. Default is None')
parser.add_argument("--bind-param",default=None,action="append",help="a parameter to impose a bound on, with corresponding --param-bound arg in respective order")
parser.add_argument("--param-bound",action="append",help="respective bounds for above params")
parser.add_argument("--ci-list",default=None,help='List for credible intervals. Default is 0.95,0.9,0.68')
parser.add_argument("--quantiles",default=None,help='List for 1d quantiles intervals. Default is 0.95,0.05')
parser.add_argument("--chi-max",default=1,type=float)
parser.add_argument("--lambda-plot-max",default=2000,type=float)
parser.add_argument("--lnL-cut",default=None,type=float)
parser.add_argument("--sigma-cut",default=0.4,type=float)
parser.add_argument("--eccentricity", action="store_true", help="Read sample files in format including eccentricity")
parser.add_argument("--matplotlib-block-defaults",action="store_true",help="Relies entirely on user to set plot options for plot styles from matplotlibrc")
parser.add_argument("--no-mod-psi",action="store_true",help="Default is to take psi mod pi. If present, does not do this")
parser.add_argument("--verbose",action='store_true',help='print matplotlibrc data')
parser.add_argument("--grid-composite-file", type=str, default=None, help="Composite file for grid data exclusively.")
parser.add_argument("--puff-composite-file", type=str, default=None, help="Composite file for puff data exclusively.")
opts=  parser.parse_args()

if opts.grid_composite_file or opts.puff_composite_file:
    if opts.composite_file:
        raise ValueError("Cannot use --composite-file with --grid-composite-file or --puff-composite-file.")
        
# Convert "None" string arguments to actual None
if opts.grid_composite_file == "None":
    opts.grid_composite_file = None
if opts.puff_composite_file == "None":
    opts.puff_composite_file = None

plt.rc('axes',unicode_minus=False)
dpi_base=200
if not(opts.matplotlib_block_defaults):
    legend_font_base=16
    rc_params = {'backend': 'ps',
             'axes.labelsize': 11,
             'axes.titlesize': 10,
             'font.size': 11,
             'legend.fontsize': legend_font_base,
             'xtick.labelsize': 11,
             'ytick.labelsize': 11,
             #'text.usetex': True,
             'font.family': 'Times New Roman'}#,
             #'font.sans-serif': ['Bitstream Vera Sans']}#,
    plt.rcParams.update(rc_params)
if opts.verbose:
    print(plt.rcParams)

if opts.posterior_file is None:
    print(" No input files ")
    import sys
    sys.exit(0)
if opts.pdf:
    fig_extension='.pdf'

truth_P_list = None
P_ref = None
truth_dat = None
if opts.truth_file:
    print(" Loading true parameters from  ", opts.truth_file)
    truth_P_list  =lalsimutils.xml_to_ChooseWaveformParams_array(opts.truth_file)
    P_ref = truth_P_list[opts.truth_event]
#    P_ref.print_params()
elif opts.truth_file_manual:
    truth_dat = np.genfromtxt(opts.truth_file_manual,names=True)

if opts.change_parameter_label:
  for name, new_str in map( lambda c: c.split("="),opts.change_parameter_label):
      if name in lalsimutils.tex_dictionary:
          lalsimutils.tex_dictionary[name] = "$"+new_str+"$"
      else:
          print(" Assigning new variable string",name,new_str)
          lalsimutils.tex_dictionary[name] = "$"+new_str+"$"  # should be able to ASSIGN NEW NAMES, not restrict

special_param_ranges = {
  'q':[0,1],
  'eta':[0,0.25],
  'a1z':[-opts.chi_max,opts.chi_max],
  'a2z':[-opts.chi_max,opts.chi_max],
  'chi_eff': [-opts.chi_max,opts.chi_max],  # this can backfire for very narrow constraints
  'lambda1':[0,4000],
  'lambda2':[0,4000],
  'chi_pavg':[0,2],
  'chi_prms':[0,2],  
  'chi_p':[0,1],
  'lambdat':[0,4000],
  'eccentricity':[0,1]
}

#mc_range deprecated by generic bind_param
#if opts.mc_range:
#    special_param_ranges['mc'] = eval(opts.mc_range)
#    print(" mc range ", special_param_ranges['mc'])
    
if opts.bind_param:
     for i,par in enumerate(opts.bind_param):
         special_param_ranges[par]=eval(opts.param_bound[i])
         print(par +" range ",special_param_ranges[par])


# Parameters
param_list = opts.parameter

# Legend
color_list=['black', 'red', 'green', 'blue','yellow','C0','C1','C2','C3']
if opts.posterior_color:
    color_list  =opts.posterior_color + color_list
else:
    color_list += len(opts.posterior_file)*['black']
linestyle_list = ['-' for k in color_list]
if opts.posterior_linestyle:
    linestyle_list = opts.posterior_linestyle + linestyle_list
#linestyle_remap_contour  = {":", 'dotted', '-'
posterior_distance_factors = np.ones(len(opts.posterior_file))
if opts.posterior_distance_factor:
    for indx in np.arange(len(opts.posterior_file)):
        posterior_distance_factors[indx] = float(opts.posterior_distance_factor[indx])

line_handles = []
corner_legend_location=None; corner_legend_prop=None
if opts.use_legend and opts.posterior_label:
    n_elem = len(opts.posterior_file)
    for indx in np.arange(n_elem):
        my_line = mlines.Line2D([],[],color=color_list[indx],linestyle=linestyle_list[indx],label=opts.posterior_label[indx])
        line_handles.append(my_line)

    corner_legend_location=(0.7, 1.0)
    corner_legend_prop = {'size':8}
# https://stackoverflow.com/questions/7125009/how-to-change-legend-size-with-matplotlib-pyplot
#params = {'legend.fontsize': 20, 'legend.linewidth': 2}
#plt.rcParams.update(params)


# Import
posterior_list = []
posteriorP_list = []
label_list = []
# Load posterior files
if opts.posterior_file:
 for fname in opts.posterior_file:
    samples = np.genfromtxt(fname,names=True,replace_space=None)  # don't replace underscores in names
    samples = standard_expand_samples(samples)
    if not(opts.no_mod_psi) and 'psi' in samples.dtype.names:
        samples['psi'] = np.mod(samples['psi'],np.pi)


    if 'chi1_perp' in samples.dtype.names:
        # impose Kerr limit, if neede
        npts = len(samples["m1"])
        indx_ok =np.arange(npts)
        chi1_squared = samples['chi1_perp']**2 + samples["a1z"]**2
        chi2_squared = samples["chi2_perp"]**2 + samples["a2z"]**2
        indx_ok = np.logical_and(chi1_squared < opts.chi_max ,chi2_squared < opts.chi_max)
        npts_out = np.sum(indx_ok)
        new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
        for name in samples.dtype.names:
            new_samples[name] = samples[name][indx_ok]
        samples = new_samples


    # Save samples
    posterior_list.append(samples)

    # Continue ... rest not used at present
    continue

    # Populate a P_list with the samples, so I can perform efficient conversion for plots
    # note only the DETECTOR frame properties are stored here
    P_list = []
    P = lalsimutils.ChooseWaveformParams()
    for indx in np.arange(len(samples["m1"])):
        P.m1 = samples["m1"][indx]*lal.MSUN_SI
        P.m2 = samples["m2"][indx]*lal.MSUN_SI
        P.s1x = samples["a1x"][indx]
        P.s1y = samples["a1y"][indx]
        P.s1z = samples["a1z"][indx]
        P.s2x = samples["a2x"][indx]
        P.s2y = samples["a2y"][indx]
        P.s2z = samples["a2z"][indx]
        if "lnL" in samples.keys():
            P.lnL = samples["lnL"][indx]   # creates a new field !
        else:
            P.lnL = -1
        # Populate other parameters as needed ...
        P_list.append(P)
    posteriorP_list.append(P_list)

for indx in np.arange(len(posterior_list)):
    samples = posterior_list[indx]
    fac = posterior_distance_factors[indx]
    if 'dist' in samples.dtype.names:
        samples["dist"]*= fac
    if 'distance' in samples.dtype.names:
        samples["distance"]*= fac

# Import
field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","lnL", "sigmaOverL", "ntot", "neff")
if opts.flag_tides_in_composite:
    if opts.flag_eos_index_in_composite:
        print(" Reading composite file, assumingtide/eos-index-based format ")
        field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","lambda1", "lambda2", "eos_indx","lnL", "sigmaOverL", "ntot", "neff")
    else:
        print(" Reading composite file, assuming tide-based format ")
        field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","lambda1", "lambda2", "lnL", "sigmaOverL", "ntot", "neff")
if opts.eccentricity:
    print(" Reading composite file, assuming eccentricity-based format ")
    field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","eccentricity", "lnL", "sigmaOverL", "ntot", "neff")
field_formats = [np.float32 for x in field_names]
composite_dtype = [ (x,float) for x in field_names] #np.dtype(names=field_names ,formats=field_formats)


def process_composite_file(file_path, opts, composite_dtype, field_names):
    """
    Load and process a composite file (standard, grid, or puff).
    
    Args:
        file_path (str): Path to the composite file.
        opts (argparse.Namespace): Parsed command-line options.
        composite_dtype (list): List of dtype tuples for the composite fields.
        field_names (tuple): Names of the fields in the composite file.

    Returns:
        tuple: (processed_samples, original_samples).
    """
    print(f"Loading composite file: {file_path}")
    
    # Load data (original samples)
    if not opts.composite_file_has_labels:
        original_samples = np.loadtxt(file_path, dtype=composite_dtype)
    else:
        original_samples = np.genfromtxt(file_path, names=True)
        original_samples = rfn.rename_fields(original_samples, {'sigmalnL': 'sigmaOverL', 'sigma_lnL': 'sigmaOverL'})

    # Start with original samples for processing
    processed_samples = original_samples.copy()

    # Remove NaN likelihoods
    if 'lnL' in processed_samples.dtype.names:
        processed_samples = processed_samples[~np.isnan(processed_samples["lnL"])]
    
    # Apply sigma cut
    if opts.sigma_cut > 0:
        sigma_vals = processed_samples["sigmaOverL"]
        good_sigma = sigma_vals < opts.sigma_cut
        processed_samples = processed_samples[good_sigma]

    # Add derived fields
    if 'm1' in processed_samples.dtype.names:
        processed_samples = add_field(processed_samples, [('mtotal', float)])
        processed_samples["mtotal"] = processed_samples["m1"] + processed_samples["m2"]
        processed_samples = add_field(processed_samples, [('q', float)])
        processed_samples["q"] = processed_samples["m2"] / processed_samples["m1"]
        processed_samples = add_field(processed_samples, [('mc', float)])
        processed_samples["mc"] = lalsimutils.mchirp(processed_samples["m1"], processed_samples["m2"])
        processed_samples = add_field(processed_samples, [('eta', float)])
        processed_samples["eta"] = lalsimutils.symRatio(processed_samples["m1"], processed_samples["m2"])
        processed_samples = add_field(processed_samples, [('chi_eff', float)])
        processed_samples["chi_eff"] = (processed_samples["m1"] * processed_samples["a1z"] + processed_samples["m2"] * processed_samples["a2z"]) / processed_samples["mtotal"]

        chi1_perp = np.sqrt(processed_samples['a1x'] ** 2 + processed_samples['a1y'] ** 2)
        chi2_perp = np.sqrt(processed_samples['a2x'] ** 2 + processed_samples['a2y'] ** 2)
        processed_samples = add_field(processed_samples, [('chi1_perp', float)])
        processed_samples['chi1_perp'] = chi1_perp
        processed_samples = add_field(processed_samples, [('chi2_perp', float)])
        processed_samples['chi2_perp'] = chi2_perp

        phi1 = np.arctan2(processed_samples['a1x'], processed_samples['a1y'])
        phi2 = np.arctan2(processed_samples['a2x'], processed_samples['a2y'])
        processed_samples = add_field(processed_samples, [('phi1', float), ('phi2', float), ('phi12', float)])
        processed_samples['phi1'] = phi1
        processed_samples['phi2'] = phi2
        processed_samples['phi12'] = phi2 - phi1

        if 'lambda1' in processed_samples.dtype.names:
            Lt, dLt = lalsimutils.tidal_lambda_tilde(processed_samples['m1'], processed_samples['m2'], processed_samples['lambda1'], processed_samples['lambda2'])
            processed_samples = add_field(processed_samples, [('LambdaTilde', float), ('DeltaLambdaTilde', float), ('lambdat', float), ('dlambdat', float)])
            processed_samples['LambdaTilde'] = processed_samples['lambdat'] = Lt
            processed_samples['DeltaLambdaTilde'] = processed_samples['dlambdat'] = dLt

    # Apply likelihood cutoff if specified
    if opts.lnL_cut and 'lnL' in processed_samples.dtype.names:
        lnL_max = np.max(processed_samples["lnL"])
        processed_samples = processed_samples[processed_samples["lnL"] > lnL_max - opts.lnL_cut]

    print(f"Processed {len(processed_samples)} samples from {file_path}")
    return processed_samples, original_samples

# Initialize lists for composite samples
composite_list = []
composite_full_list = []
grid_composite_list = []
grid_composite_full_list = []
puff_composite_list = []
puff_composite_full_list = []

# Process grid composite file
if opts.grid_composite_file is not None:  
    grid_samples, grid_samples_original = process_composite_file(opts.grid_composite_file, opts, composite_dtype, field_names)
    
    grid_composite_list.append(grid_samples)
    grid_composite_full_list.append(grid_samples_original)
    

# Process puff composite file
if opts.puff_composite_file is not None:
    puff_samples, puff_samples_original = process_composite_file(opts.puff_composite_file, opts, composite_dtype, field_names)
    
    puff_composite_list.append(puff_samples)
    puff_composite_full_list.append(puff_samples_original)

# Process standard composite file (if no grid or puff file is specified)
if opts.composite_file is not None:
    # Ensure opts.composite_file is a single string, not a list
    composite_file_path = opts.composite_file[0] if isinstance(opts.composite_file, list) else opts.composite_file

    processed_samples, original_samples = process_composite_file(composite_file_path, opts,composite_dtype, field_names)
    composite_list.append(processed_samples)
    composite_full_list.append(original_samples)


## Plot posterior files

CIs = [0.95,0.9, 0.68]
if opts.ci_list:
    CIs = eval(opts.ci_list)  # force creation
quantiles_1d = [0.05,0.95]
if opts.quantiles:
    quantiles_1d=eval(opts.quantiles)

# Generate labels
if opts.parameter_log_scale is None:
    opts.parameter_log_scale = []
labels_tex = render_coordinates(opts.parameter,logparams=opts.parameter_log_scale)#map(lambda x: tex_dictionary[x], coord_names)

fig_base= None
# Create figure workspace for 1d plots
fig_1d_list = []
fig_1d_list_cum = []
#fig_1d_list_ids = []
if opts.plot_1d_extra:
    for indx in np.arange(len(opts.parameter))+5:
        fig_1d_list.append(plt.figure(indx))
        fig_1d_list_cum.append(plt.figure(indx+len(opts.parameter)))
#        fig_1d_list_ids.append(indx)
    plt.figure(1)


# Find parameter ranges
x_range = {}
range_list = []
if opts.posterior_file:
 for param in opts.parameter:
    xmax_list = []
    xmin_list = []
    for indx in np.arange(len(posterior_list)):
        dat_here = None
        samples = posterior_list[indx]
        if param in samples.dtype.names:
            dat_here = samples[param]
        else:
            dat_here = extract_combination_from_LI(samples, param)
        if param in opts.parameter_log_scale:
            indx_ok = dat_here > 0
            dat_here= np.log10(dat_here[indx_ok])
        if len(dat_here) < 1:
            print(" Failed to etract data ", param,  " from ", opts.posterior_file[indx])

        # extend the limits, so we have *extremal* limits 
        xmax_list.append(np.max(dat_here))
        xmin_list.append(np.min(dat_here))
    x_range[param] = np.array([np.min(xmin_list), np.max(xmax_list)])  # give a small buffer
#    if param == 'chi_eff':
#        x_range[param] -= 0.1*np.sign([-1,1])*(x_range[param]+np.array([-1,1]))
            
    if param in special_param_ranges:
        x_range[param] = special_param_ranges[param]

    if param in ['lambda1', 'lambda2', 'lambdat']:
        x_range[param][1] = opts.lambda_plot_max

    range_list.append(x_range[param])
    print(param, x_range[param])


my_cmap_values=None
for pIndex in np.arange(len(posterior_list)):
    samples = posterior_list[pIndex]
    sample_names = samples.dtype.names; sample_ref_name  = sample_names[0]
    # Create data for corner plot
    dat_mass = np.zeros( (len(list(samples[sample_ref_name])), len(list(labels_tex))) )
    my_cmap_values = color_list[pIndex]
    plot_range_list = []
    smooth_list =[]
    truths_here= None
    if opts.truth_file or opts.truth_file_manual:
        truths_here = np.zeros(len(opts.parameter))
    for indx in np.arange(len(opts.parameter)):
        param = opts.parameter[indx]
        if param in samples.dtype.names:
            dat_mass[:,indx] = samples[param]
        else:
            dat_mass[:,indx] = extract_combination_from_LI(samples, param)

        if param in opts.parameter_log_scale:
            dat_mass[:,indx] = np.log10(dat_mass[:,indx])

        # Parameter ranges (duplicate)
        dat_here = np.array(dat_mass[:,indx])  # force copy ! I need to sort
        weights = np.ones(len(dat_here))*1.0/len(dat_here)
        if 'weights' in samples.dtype.names:
            weights = samples['weights']
        indx_sort= dat_here.argsort()
        dat_here = dat_here[indx_sort]
        weights =weights[indx_sort]
#
        dat_here.sort() # sort it
        xmin, xmax = x_range[param]
#            xmin = np.min([np.min( posterior_list[x][param]) for x in np.arange(len(posterior_list)) ]) # loop over all
        xmin  = np.min([xmin, np.mean(dat_here)  -4*np.std(dat_here)])
        xmax = np.max([xmax, np.mean(dat_here)  +4*np.std(dat_here)])
#            xmax  = np.max(dat_here)
        if param in special_param_ranges:
                xmin,xmax = special_param_ranges[param]
        plot_range_list.append((xmin,xmax))

        # smoothing list
        smooth_list.append(np.std(dat_here)/np.power(len(dat_here), 1./3))
        
        # truths
        if opts.truth_file_manual:
            truths_here[indx] = truth_dat[param]
        if opts.truth_file:
            param_to_extract = param
            if param in remap_LI_to_ILE.keys():
                param_to_extract  = remap_LI_to_ILE[param]
            if param in eos_param_names:
                continue
            if param == 'time':
                truths_here[indx] = P_ref.tref
                continue
            truths_here[indx] = P_ref.extract_param(param_to_extract)
            if param in [ 'mc', 'm1', 'm2', 'mtotal']:
                truths_here[indx] = truths_here[indx]/lal.MSUN_SI
            if param in ['dist', 'distance']:
                truths_here[indx] = truths_here[indx]/lal.PC_SI/1e6
#            print param, truths_here[indx]

        # if 1d plots needed, make them
        if opts.plot_1d_extra:
            range_here = range_list[indx]
            # 1d PDF
            # Set range based on observed results in ALL sets of samples, by default
            fig =fig_1d_list[indx]
            ax = fig.gca()
            ax.set_xlabel(labels_tex[indx])
            ax.set_ylabel('$dP/d'+labels_tex[indx].replace('$','')+"$")
            try:
                my_kde = bounded_kde.BoundedKDE(dat_here,low=xmin,high=xmax)
                xvals = np.linspace(range_here[0],range_here[1],1000)
                yvals = my_kde.evaluate(xvals)
                ax.plot(xvals,yvals,color=my_cmap_values,linestyle= linestyle_list[pIndex])
                if opts.truth_file:
                    ax.axvline(truths_here[indx], color='k',linestyle='dashed')
            except:
                print(" Failed to plot 1d KDE for ", labels_tex[indx])

            # 1d CDF
            fig =fig_1d_list_cum[indx]
            ax = fig.gca()
            ax.set_xlabel(labels_tex[indx])
            ax.set_ylabel('$P(<'+labels_tex[indx].replace('$','')+")$")
            xvals = dat_here
            #yvals = np.arange(len(dat_here))*1.0/len(dat_here)
            yvals = np.cumsum(weights)
            yvals = yvals/yvals[-1]
            ax.plot(xvals,yvals,color=my_cmap_values,linestyle= linestyle_list[pIndex] )
            if opts.truth_file:
                ax.axvline(truths_here[indx], color='k',linestyle='dashed')
            ax.set_xlim(xmin,xmax)

    # Add weight columns (unsorted) for overall unsorted plot
    weights = np.ones(len(dat_mass))*1.0/len(dat_mass)
    if 'weights' in samples.dtype.names:
        weights= samples['weights']
        weights = weights/np.sum(weights)
    # plot corner
#    smooth=smooth_list
    smooth1d=None
#    if opts.use_smooth_1d:
#        smooth1d=smooth_list
#        print smooth1d
    fig_base = corner.corner(dat_mass,smooth1d=smooth1d, range=range_list,weights=weights, labels=labels_tex, quantiles=quantiles_1d, plot_datapoints=False, plot_density=False, no_fill_contours=True, contours=True, levels=CIs,fig=fig_base,color=my_cmap_values ,hist_kwargs={'linestyle': linestyle_list[pIndex]}, linestyle=linestyle_list[pIndex],contour_kwargs={'linestyles':linestyle_list[pIndex]},truths=truths_here)


if opts.plot_1d_extra:
    for indx in np.arange(len(opts.parameter)):
        fig = fig_1d_list[indx]
        param = opts.parameter[indx]
        ax = fig.gca()
        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
        if opts.use_legend:
            ax.legend(handles=line_handles, bbox_to_anchor=corner_legend_location, prop=corner_legend_prop,loc=2) 
        fig.savefig(param+fig_extension,dpi=dpi_base)

        fig = fig_1d_list_cum[indx]
        param = opts.parameter[indx]
        ax = fig.gca()
        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
        if opts.use_legend:
            ax.legend(handles=line_handles, prop=corner_legend_prop, loc=4) # bbox_to_anchor=corner_legend_location, 
        fig.savefig(param+"_cum"+fig_extension,dpi=dpi_base)


composite_sources = [
    ("standard", composite_list, composite_full_list),
    ("grid", grid_composite_list, grid_composite_full_list),
    ("puff", puff_composite_list, puff_composite_full_list),
]

# Pre-compute the global lnL min and max
global_lnL_min = float("inf")
global_lnL_max = float("-inf")

for _, source_list, _ in composite_sources:
    if not source_list: # skip empty lists
        continue
    for source in source_list:
        if "lnL" in source.dtype.names:
            lnL_values = source["lnL"]
            global_lnL_min = min(global_lnL_min, lnL_values.min())
            global_lnL_max = max(global_lnL_max, lnL_values.max())

# Ensure global y_span is meaningful
if global_lnL_max > global_lnL_min:
    global_y_span = global_lnL_max - global_lnL_min
else:
    print('lnL evaluation error - falling back to dummy')
    global_y_span = 1.0  # Avoid division by zero; fallback to dummy span

for source_name, source_list, source_full_list in composite_sources:
    if not source_list:  # Skip empty lists
        continue
    
    print(f"Processing {source_name} composite data...")
    for pIndex in range(len(source_list)):
        samples = source_list[pIndex]
        samples_orig = source_full_list[pIndex]
        samples_ref_name = samples.dtype.names[0]
        samples_orig_ref_name = samples_orig.dtype.names[0]

        # Create data for corner plot
        dat_mass = np.zeros((len(np.atleast_1d(samples[samples_ref_name])), len(labels_tex)))
        dat_mass_orig = np.zeros((len(np.atleast_1d(samples_orig[samples_orig_ref_name])), len(labels_tex)))
        cm = matplotlib.colormaps["rainbow"]  # 'RdYlBu_r'

        # Use global y_span for color scale
        if "lnL" in samples.dtype.names:
            lnL = samples["lnL"]
            indx_sorted = lnL.argsort()
            if len(lnL) < 1:
                print(f"Failed to retrieve lnL for {source_name} composite file.")
                continue

            my_cmap_values = cm((lnL - global_lnL_min) / global_y_span)
        else:
            my_cmap_values = cm(np.ones(len(np.atleast_1d(samples[samples_ref_name]))))
            indx_sorted = np.arange(len(np.atleast_1d(samples[samples_ref_name])))

        truths_here = None
        if opts.truth_file:
            truths_here = np.zeros(len(opts.parameter))
        for indx in np.arange(len(opts.parameter)):
            param = opts.parameter[indx]
            if param in field_names:
                dat_mass[:, indx] = samples[param]
                dat_mass_orig[:, indx] = samples_orig[param]
            else:
                print(f"Trying alternative access for {param}")
                dat_mass[:, indx] = extract_combination_from_LI(samples, param)
                dat_mass_orig[:, indx] = extract_combination_from_LI(samples_orig, param)
            
            # Truths
            if opts.truth_file:
                param_to_extract = param
                if param in remap_LI_to_ILE.keys():
                    param_to_extract = remap_LI_to_ILE[param]
                truths_here[indx] = P_ref.extract_param(param_to_extract)
                if param in ["mc", "m1", "m2", "mtotal"]:
                    truths_here[indx] = truths_here[indx] / lal.MSUN_SI

        # Fix ranges
        if range_list == []:
            range_list = None

        # Reverse order ... make sure largest plotted last
        dat_mass = dat_mass[indx_sorted]  # Sort by lnL
        my_cmap_values = my_cmap_values[indx_sorted]
        
        # point style
        point_style_def = {"color": my_cmap_values, "s": 1}
        point_style_puff = {"color": my_cmap_values, "s": 2, "marker": "+"}
        point_style_gray = {"color": "0.5", "s": 1}
        point_style_gray_puff = {"color": "0.5", "s": 2, "marker":"+"}
        

        # Grayscale, using all points
        if opts.use_all_composite_but_grayscale:
            if source_name =="puff":
                fig_base = our_corner.corner(
                    dat_mass_orig,
                    range=range_list,
                    plot_datapoints=True,
                    weights=np.ones(len(dat_mass_orig)) * 1.0 / len(dat_mass_orig),
                    plot_density=False,
                    no_fill_contours=True,
                    plot_contours=False,
                    contours=False,
                    levels=None,
                    fig=fig_base,
                    data_kwargs=point_style_gray_puff,
                )
            else:
                fig_base = our_corner.corner(
                    dat_mass_orig,
                    range=range_list,
                    plot_datapoints=True,
                    weights=np.ones(len(dat_mass_orig)) * 1.0 / len(dat_mass_orig),
                    plot_density=False,
                    no_fill_contours=True,
                    plot_contours=False,
                    contours=False,
                    levels=None,
                    fig=fig_base,
                    data_kwargs=point_style_gray,
                )

        # Color scale with colored points
        if source_name == "puff":
            fig_base = our_corner.corner(
                dat_mass,
                range=range_list,
                plot_datapoints=True,
                weights=np.ones(len(dat_mass)) * 1.0 / len(dat_mass),
                plot_density=False,
                no_fill_contours=True,
                plot_contours=False,
                contours=False,
                levels=None,
                fig=fig_base,
                data_kwargs=point_style_puff,
                truths=truths_here,
            )
        else:
            fig_base = our_corner.corner(
                dat_mass,
                range=range_list,
                plot_datapoints=True,
                weights=np.ones(len(dat_mass)) * 1.0 / len(dat_mass),
                plot_density=False,
                no_fill_contours=True,
                plot_contours=False,
                contours=False,
                levels=None,
                fig=fig_base,
                data_kwargs=point_style_def,
                truths=truths_here,
            )
            


    # Create colorbar mappable
#    ax=plt.figure().gca()
#    ax.contourf(lnL, cm)

if opts.use_legend and opts.posterior_label:
    plt.legend(handles=line_handles, bbox_to_anchor=corner_legend_location, prop=corner_legend_prop,loc=4)
#plt.colorbar()  # Will fail, because colors not applied

# title
if opts.use_title:
    print(" Addding title ", opts.use_title)
    plt.title(opts.use_title)

param_postfix = "_".join(opts.parameter)
res_base = len(opts.parameter)*dpi_base
if not(opts.matplotlib_block_defaults):
    matplotlib.rcParams.update({'font.size': 11+int(len(opts.parameter)), 'legend.fontsize': legend_font_base+int(1.3*len(opts.parameter))})   # increase font size if I have more panels, to keep similar aspect
plt.savefig("corner_"+param_postfix+fig_extension,dpi=res_base)        # use more resolution, to make sure each image remains of consistent quality
