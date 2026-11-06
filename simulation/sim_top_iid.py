import zarr
import numpy as np
import argparse
import os
import struct
import random

# This script simulates an iid phenotype on top of real methylation data
# INPUT:
#       - Parameters for simulation
#       - adjusted and standardized methylation data stored in zarr files (one file per chromosome)
#       - output file path
# OUTPUT:
#       - .bin files for train and test subsets. Here we store design matrix of 8 byte doubles.
#       - .bin file of true signals stored as 8 byte doubles.
#       - simulated phenotype file .phen in plink format

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-zarr", "--zarr", help = "Path to zarr files")
parser.add_argument("-out", "--out", help = "Output file path")
parser.add_argument("-phen", "--phen", help = "Phenotype name", default="sim")
parser.add_argument("-dataset", "--dataset", help = "Dataset name")
parser.add_argument("-h2", "--h2", help = "Heritability", default=0.8)
parser.add_argument("-lam", "--lam", help = "Sparsity (lambda)", default=0.01)
parser.add_argument("-run", "--run", help = "Run number", default=0)
parser.add_argument("-ratio", "--ratio", help = "Proportion of train data", default=0.9)
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-N", "--N", help = "Number of samples")
args = parser.parse_args()

# Input parameters
zarr_fpath = args.zarr
out_fpath = args.out
phen_name = args.phen
dataset_name = args.dataset
lam = float(args.lam)
h2 = float(args.h2)
ratio = float(args.ratio)
run = int(args.run)
M = int(args.M) # number of markers
N = int(args.N) # number of samples

# Suffix for simulation file names
sub = "h2_%d_lam_%d_run_%d" % (h2*100, lam*100, run)

# File names for simulation
fname = "%s_%s_%s" % (dataset_name, phen_name, sub)
fname_train = "%s_train_%s_%s" % (dataset_name, phen_name, sub)
fname_test = "%s_test_%s_%s" % (dataset_name, phen_name, sub)

# Generate random mask for sampling train and test subsets
msk = np.random.rand(N) < ratio
N_train = sum(msk)
N_test = sum(~msk)
print("Number of train samples:", N_train, flush=True)
print("Number of test samples:", N_test, flush=True)

# Save train test indicator mask
np.savetxt(os.path.join(out_fpath, fname + ".msk"), msk)

# Save .dim files. Information on number of samples and markers
dimf_train = open(os.path.join(out_fpath, fname_train + ".dim"), 'w')
dimf_test = open(os.path.join(out_fpath, fname_test + ".dim"), 'w')
dimf_train.write("%d %d" % (N_train, M))
dimf_test.write("%d %d" % (N_test, M))
dimf_train.close()
dimf_test.close()

# List of zarr files in directory (sorted so the marker order is reproducible)
files = sorted(os.listdir(zarr_fpath))

# Number of causal markers
cm = int(M * lam)

# beta variance
bvar = 1 / cm

# indices of causal markers
idx = random.sample(range(M), cm)

# true signals beta
beta = np.zeros(M)
beta[idx] = np.random.normal(0,np.sqrt(bvar),cm)
print("Var(beta) =", bvar, flush=True)

# Save true signals to file
beta_true_binf = open(os.path.join(out_fpath, fname + "_beta_true.bin"), "wb")
beta_true_binf.write(struct.pack(str(M)+'d', *beta.squeeze()))
beta_true_binf.close()

# Vector for storing X @ beta
g = np.zeros(N)

# Total number of loaded markers
Mtot = 0

# Open output binary files for design matrix
train_binf = open(os.path.join(out_fpath, fname_train + ".bin"), "wb")
test_binf = open(os.path.join(out_fpath, fname_test + ".bin"), "wb")

# For all zarr files (one file per chromosome)
for i,f in enumerate(files):

    # zarr file name
    print("Processing file %s" % f, flush=True)

    # Loaded chunk of data (one chromosome)
    store = zarr.open(os.path.join(zarr_fpath, f))

    # Check dimensions
    Ni = np.shape(store)[0]
    Mi = np.shape(store)[1]
    if N != Ni:
        raise Exception("Number of samples in zarr file and specified do not match!")

    # Mask chunk of data (store), transpose, flatten and write to binary file as a vector. Coded as one long command for memory efficiency
    train_binf.write(struct.pack(str(N_train*Mi)+'d', *np.array(store)[msk,:].transpose().ravel().squeeze())) # (N, Mi) -> (N_train, Mi) -> (Mi, N_train) -> (Mi * N_train)
    test_binf.write(struct.pack(str(N_test*Mi)+'d', *np.array(store)[~msk,:].transpose().ravel().squeeze())) # (N, Mi) -> (N_test, Mi) -> (Mi, N_test) -> (Mi * N_test)

    # X @ beta for current chromosome
    g += np.matmul(np.array(store), beta[Mtot:(Mtot+Mi)])
    Mtot += Mi

    # free memory
    del store

# Close binary files
train_binf.close()
test_binf.close()

if Mtot != M:
    raise Exception("Number of markers in zarr files and specified do not match!")
print("Total number of loaded markers:", Mtot, flush=True)

# Variance of noise based on specified heritability
evar = 1.0 / h2 - 1.0

# Add noise to g
y = g + np.random.normal(0, np.sqrt(evar), N)

print("Var(g) =", np.var(g), flush=True)
print("Var(y) =", np.var(y), flush=True)
print("h2 =", np.var(g) / np.var(y), flush=True)

# Standardize phenotype
y = (y - np.mean(y)) / np.std(y)
print("After standardization: Var(y) =", np.var(y), flush=True)

# Open output files for phenotype
phenf_test = open(os.path.join(out_fpath, fname_test + ".phen"), "w")
phenf_train = open(os.path.join(out_fpath, fname_train + ".phen"), "w")

for i, pheno in enumerate(y):

    line = "%d %d %0.10f\n" % (i, i, pheno)

    # train test splits
    if msk[i]:
        phenf_train.write(line)
    else:
        phenf_test.write(line)

# Close output files
phenf_train.close()
phenf_test.close()
