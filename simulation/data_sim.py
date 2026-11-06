import numpy as np
import argparse
import os
import struct
import random

print("---- Simulating example i.i.d. data ----\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-out_dir", "--out-dir", help = "Output directory")
parser.add_argument("-out_name", "--out-name", help="Output file name", default="example")
parser.add_argument("-N", "--N", help="Number of individuals", default=1000)
parser.add_argument("-M", "--M", help="Number of markers", default=2000)
parser.add_argument("-lam", "--lam", help="Sparsity", default=0.1)
parser.add_argument("-h2", "--h2", help="Heritability", default=0.8)
args = parser.parse_args()

out_name = args.out_name
out_dir = args.out_dir
N = int(args.N)
M = int(args.M)
lam = float(args.lam)
h2 = float(args.h2)

print("Input arguments:")
print("--out_dir", out_dir)
print("--out_name", out_name)
print("--N", N)
print("--M", M)
print("--lam", lam)
print("--h2", h2, flush=True)

print("\n...Simulating design matrix", flush=True)
X = np.random.normal(0,1,N*M).reshape((N,M))

print("\n...Simulating marker effects", flush=True)
CM = int(M * lam) # number of causal markers
sigma2 = h2 / CM
idx = random.sample(range(M), CM)
beta = np.zeros(M)
beta[idx] = np.random.normal(0.0, np.sqrt(sigma2), CM)
print("sigma2 = ", sigma2, flush=True)

print("\n...Computing outcome variable", flush=True)
g = np.matmul(X,beta)
y = g + np.random.normal(0, np.sqrt(1 - h2),  N) # adding Gaussian noise

print("Var(y) = ", np.var(y))
print("Var(g) = ", np.var(g))
print("h2 = ", np.var(g) / np.var(y), flush=True)

# Saving data in binary format
print("\n...Saving design matrix to bin file")
bin_fpath = os.path.join(out_dir, "%s.bin" % out_name)
print(bin_fpath, flush=True)
binf = open(bin_fpath, "wb")
b = struct.pack(str(N*M)+'d', *X.transpose().ravel().squeeze())
binf.write(b)
binf.close()

# Saving phenotype data
print("\n...Saving outcome to .phen file")
phen_fpath = os.path.join(out_dir, "%s.phen" % out_name)
print(phen_fpath, flush=True)
phenf = open(phen_fpath, "w")
for i, pheno in enumerate(y):
    line = "%d %d %0.10f\n" % (i, i, pheno)
    phenf.write(line)
phenf.close()

# Save true signals to file
print("\n...Saving true signals to bin file")
ts_bin_fpath = os.path.join(out_dir, "%s_ts.bin" % out_name)
print(ts_bin_fpath, flush=True)
ts_binf = open(ts_bin_fpath, "wb")
ts_binf.write(struct.pack(str(M)+'d', *beta.squeeze()))
ts_binf.close()
