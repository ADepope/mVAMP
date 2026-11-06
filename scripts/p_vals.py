import numpy as np
import argparse
import struct
from scipy.stats import norm
import csv
import os
from array import array

# This script calculates state evolution p-values from the r1 vector
print("----- Computing mVAMP p-values -----")
print("\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-out_name", "--out-name", help = "Output file name")
parser.add_argument("-csv_params", "--csv-params", help = "Path to parameters csv file")
parser.add_argument("-r1_file", "--r1-file", help = "Path to r1 binary file")
parser.add_argument("-it", "--it", help = "Target iteration", default=35)
parser.add_argument("-th", "--th", help = "P-values threshold", default=0.05)
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-N", "--N", help = "Number of samples")
args = parser.parse_args()

csv_params_fpath = args.csv_params
out_name = args.out_name
r1_fpath = args.r1_file
Mt = int(args.M)
N = int(args.N)
it = int(args.it)
th = float(args.th)

pvals_thr = th / Mt

# Get basename of train csv
csv_base = os.path.basename(csv_params_fpath)
csv_base = csv_base.split('_params')[0]
csv_dirpath = os.path.dirname(csv_params_fpath)

# ----------- Reading _params.csv file ----------- #
csv_params_file = open(csv_params_fpath, newline='', encoding='utf-8')
csv_params_reader = csv.reader((row.replace('\0', '') for row in csv_params_file), delimiter=',')
next(csv_params_reader, None) # Skip header

gam1 = []
for row in csv_params_reader:
    # train output csv file structure:
    #| iteration | alpha1 | gam1 | alpha2 | gam2 | gamw |
    gam1.append(float(row[2]))
gam1 = np.array(gam1)

print("...Reading file")
print(r1_fpath)
print("\n", flush=True)
r1_binfile = open(r1_fpath, "rb")
buffer = r1_binfile.read(Mt*8)
r1 = struct.unpack(str(Mt)+'d', buffer)

pvals = np.zeros(Mt)
for i in range(Mt):
    pvals[i] = norm.cdf(x=0, loc=r1[i], scale=np.sqrt(1 / (gam1[it-1] * N)))
    if r1[i] <= 0:
        pvals[i] = 1 - pvals[i]

outf = os.path.join(csv_dirpath, out_name+'.bin')
print("...Saving p-values to file")
print(outf)
print("\n", flush=True)
output_file = open(outf, 'wb')
float_array = array('d', pvals)
float_array.tofile(output_file)
output_file.close()

print("...Printing results")
print("-"*45)
print("| %3s | %8s | %24s |" % ("It.", "gam1", "Number of causal markers"))
print("-"*45)
print("| %3d | %8.4f | %24d |" % (it, gam1[it-1], sum(pvals <= pvals_thr)), flush=True)
print("-"*45)
print("\n", flush=True)
