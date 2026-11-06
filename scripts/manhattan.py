import numpy as np
import argparse
import struct
import os
import matplotlib.pyplot as plt
import pandas as pd
import csv

# This script visualize manhattan plot
print("---------- Association testing for mVAMP ----------")
print("\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-pval", "--pval", help = "Path to pvals bin file")
parser.add_argument("-probes", "--probes", help = "Path to probes file")
parser.add_argument("-out_name", "--out-name", help = "Output file name")
parser.add_argument("-trait", "--trait", help = "Trait name", default="")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-th", "--th", help = "P-values threshold", default=0.05)
args = parser.parse_args()
pvalfile = args.pval
probes_fpath = args.probes
out_name = args.out_name
trait = args.trait
M = int(args.M)
th = float(args.th)

print("Input arguments:")
print("--pval", pvalfile)
print("--probes", probes_fpath)
print("--out-name", out_name)
print("--trait", trait)
print("--M", M)
print("\n", flush=True)

dirpath = os.path.dirname(pvalfile)

print("...Reading p-values from the file")
print(pvalfile)
print("\n", flush=True)

# Get probes
Mt = 0
Mchr = []
probes = []
chrs = []
for chr in range(22):
    probes_df = pd.read_csv(probes_fpath+str(chr+1)+".txt", header=None)
    probesj = probes_df[0].to_list()
    Mj = len(probesj)
    for p in probesj:
        probes.append(p)
        chrs.append(chr+1)
    Mt+=Mj
    Mchr.append(Mj)

if Mt != M:
    raise Exception("Number of markers specified %d is not same as in probes file %d!" % (M, Mt))

probes = np.array(probes)
chrs = np.array(chrs)

pval_th = th / M

#plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(12,8), dpi=300)
plt.plot(np.array([0, M]), np.array([-np.log10(pval_th), -np.log10(pval_th)]), "k--")
plt.xlabel('Chromosome', fontsize=22)
plt.ylabel(r'$-log_{10}(p)$', fontsize=22)
plt.title("mVAMP - %s" % (trait), fontsize=26)

f = open(pvalfile, "rb")
buffer = f.read(M * 8)
pvals = struct.unpack(str(M)+'d', buffer)
pvals = np.array(pvals)

#Saturation by min value
pvals_sat = pvals
pvals_sat[pvals_sat <= 0] = min(pvals_sat[pvals_sat > 0])

chrs_c = []
chrs_ticks = []
for chr in range(22):
    js = sum(Mchr[:chr])
    je = sum(Mchr[:chr+1])
    pvals_chr = pvals_sat[js:je]
    plt.scatter(x=np.arange(js,je), y=-np.log10(pvals_chr), s=6)
    
    if chr % 2 == 0:
        chrs_ticks.append("")
    else:
        chrs_ticks.append(str(chr+1))
    chrs_c.append(js + np.round(Mchr[chr] / 2))

plt.xticks(chrs_c, chrs_ticks, fontsize=15)
plt.yticks(fontsize=15)

fout = os.path.join(dirpath, out_name+'.png')
print("...Saving manhattan figure to file")
print(fout)
print("\n", flush=True)
plt.savefig(fout)

header = "| Number of associations |"
line = "_" * len(header)
row = "| %22d |" % sum(pvals <= pval_th)
print(line)
print(header)
print(line)
print(row)
print(line)
print("\n", flush=True)

fout = os.path.join(dirpath, out_name+'.csv')
print("...Saving metrics to CSV file")
print(fout)
print("\n", flush=True)

csv_file = open(fout, 'w', newline="")
csv_writer = csv.writer(csv_file, delimiter='\t')
row = [sum(pvals <= pval_th)]
csv_writer.writerow(row)
csv_file.close()