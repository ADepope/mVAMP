import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import csv

print("---- Ploting mVAMP metrics and parameters ----")
print("\n",flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-csv_metrics", "--csv-metrics", help = "Path to metrics csv file")
parser.add_argument("-csv_test", "--csv-test", help = "Path to test csv file")
parser.add_argument("-csv_params", "--csv-params", help = "Path to parameters csv file")
parser.add_argument("-csv_prior", "--csv-prior", help = "Path to prior mixtures csv file")
parser.add_argument("-iterations", "--iterations", help = "Number of iterations", default=35)

args = parser.parse_args()
csv_params_fpath = args.csv_params
csv_metrics_fpath = args.csv_metrics
csv_test_fpath = args.csv_test
csv_prior_fpath = args.csv_prior
it = int(args.iterations)

print("Input arguments:")
print("--csv-metrics", csv_metrics_fpath)
print("--csv-test", csv_test_fpath)
print("--csv-params", csv_params_fpath)
print("--csv-prior", csv_prior_fpath)
print("--iterations", it)
print("\n", flush=True)

# Get basename of train csv
base = os.path.basename(csv_metrics_fpath)
base = base.split('.')[0]
dirpath = os.path.dirname(csv_metrics_fpath)

# ----------- Reading _test.csv file ----------- #
csv_test_file = open(csv_test_fpath, newline='', encoding='utf-8')
csv_test_reader = csv.reader((row.replace('\0', '') for row in csv_test_file), delimiter=',')
next(csv_test_reader, None) # Skip header
r2_test = []
corr2_test = []

for row in csv_test_reader:
    # test output csv file structure: 
    #| iteration | R2 | Corr2 |
    r2_test.append(float(row[1]))
    corr2_test.append(float(row[2]))

r2_test = np.array(r2_test)
corr2_test = np.array(corr2_test)

# ----------- Reading _metrics.csv file ----------- #

csv_metrics_file = open(csv_metrics_fpath, newline='', encoding='utf-8')
csv_metrics_reader = csv.reader((row.replace('\0', '') for row in csv_metrics_file), delimiter=',')
next(csv_metrics_reader, None) # Skip header

r2_denoising = []
r2_lmmse = []
corr_train = []
for row in csv_metrics_reader:
    # train output csv file structure: 
    #| iteration | R2 | x1_corr | R2 | x2_corr |
    r2_denoising.append(float(row[1]))
    r2_lmmse.append(float(row[3]))
    corr_train.append(float(row[2]))

r2_denoising = np.array(r2_denoising)
r2_lmmse = np.array(r2_lmmse)
corr_train = np.array(corr_train)

# ----------- Reading _params.csv file ----------- #
csv_params_file = open(csv_params_fpath, newline='', encoding='utf-8')
csv_params_reader = csv.reader((row.replace('\0', '') for row in csv_params_file), delimiter=',')
next(csv_params_reader, None) # Skip header

gam1 = []
gamw = []
for row in csv_params_reader:
    # train output csv file structure: 
    #| iteration | alpha1 | gam1 | alpha2 | gam2 | gamw |
    gam1.append(float(row[2]))
    gamw.append(float(row[5]))
gam1 = np.array(gam1)
gamw = np.array(gamw)

# ----------- Reading _prior.csv file ----------- #
csv_prior_file = open(csv_prior_fpath, newline='', encoding='utf-8')
csv_prior_reader = csv.reader((row.replace('\0', '') for row in csv_prior_file), delimiter=',')
next(csv_prior_reader, None) # Skip header

lam = []
for row in csv_prior_reader:
    # train output csv file structure: 
    lam.append( 1 - float(row[2]))

lam = np.array(lam)

# ------------- Ploting metrics and parameters ------------ #
fig, ax = plt.subplots(3, figsize=(12, 10), dpi=300)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
color = colors[0]

x = np.arange(1, it+1, 1) # x-axis (start,stop,step)

fig.suptitle(base)

ax[0].plot(x,r2_denoising[:it], color=color, linestyle='--', marker=".", label="Denoising")
ax[0].plot(x,r2_lmmse[:it], color=color, linestyle=':', marker=".", label="LMMSE")
ax[0].plot(x,r2_test[:it], color=color, linestyle='-', marker=".", label="Test")
ax[0].xaxis.set_ticks(x)
ax[0].set_ylim([0,1])
ax[0].set_ylabel("R2")
ax[0].legend()

ax[1].plot(x,gamw[:it], color=color, marker=".", label="gamw")
ax[1].xaxis.set_ticks(x)
ax[1].set_ylabel("gamw")

ax[2].plot(x,gam1[:it], color=color, marker=".", label="gam1")
ax[2].xaxis.set_ticks(x)
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("gam1")

# Save figure
outf = os.path.join(dirpath, base+'.png')
fig.savefig(outf)
print("...saving figure to file", outf)
print("\n", flush=True)

h2 = 1 - (1 / gamw[it-1])

print("...Printing final results")
header = "| %10s | %13s | %13s | %13s | %13s | %13s | %13s | %13s |" % ("Iteration", "R2_test", "Corr2_test", "R2_denoising", "R2_lmmse", "gam1", "gamw", "h2")
line = "-"*len(header)
row = "| %10d | %13.4f | %13.4f | %13.4f | %13.4f | %13.4f | %13.4f | %13.4f |" % (it, r2_test[it-1], corr2_test[it-1], r2_denoising[it-1], r2_lmmse[it-1], gam1[it-1], gamw[it-1], h2)
print(line)
print(header)
print(line)
print(row)
print(line)
print("\n", flush=True)