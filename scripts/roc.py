import numpy as np
import argparse
import struct
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

eps = 1e-32

# This script visualize ROC curve
print("...Ploting ROC curve for mVAMP", flush=True)
print("\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-pval", "--pval", help = "Path to binary file storing p values")
parser.add_argument("-true_signal", "--true-signal", help = "Path to binary file storing true signals")
parser.add_argument("-out_name", "--out-name", help = "Output file name")
parser.add_argument("-it", "--it", help = "Target iteration", default=35)
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-th", "--th", help = "P-values threshold", default=0.05)
args = parser.parse_args()

pvalfile = args.pval
true_signal_fpath = args.true_signal
out_name = args.out_name
M = int(args.M)
it = int(args.it)
th = float(args.th)

print("Input arguments:")
print("--pval", pvalfile)
print("--true-signal", true_signal_fpath)
print("--out-name", out_name)
print("--M", M)
print("--it", it)
print("--th", th)
print("\n", flush=True)

dirpath = os.path.dirname(pvalfile)

f = open(true_signal_fpath, "rb")
buffer = f.read(M * 8)
beta = struct.unpack(str(M)+'d', buffer)
beta = np.array(beta)

true = np.zeros(M)
true[np.abs(beta) > 0] = 1

plt.plot(np.array([0, 1]), np.array([0, 1]), "k--")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

f = open(pvalfile, "rb")
buffer = f.read(M * 8)
pvals = struct.unpack(str(M)+'d', buffer)
pvals = np.array(pvals)

fprs, tprs, thresholds = roc_curve(true, 1 - pvals)
area = auc(fprs, tprs)
plt.plot(fprs, tprs, label=it)

pval_th = 0.05 / M
est = np.zeros(M)
est[pvals < pval_th] = 1

tn, fp, fn, tp = confusion_matrix(true, est).ravel()
fdr = fp / (fp + tp + eps)
tpr = tp / (tp + fn + eps)

plt.legend()

# Save figure
print("...saving ROC figure to file")
print(os.path.join(dirpath, out_name+'.png'), flush=True)
print("\n", flush=True)
plt.savefig(os.path.join(dirpath, out_name+'.png'))

print("-"*62)
print("| %3s | %25s | %6s | %6s | %6s |" % ("It.", "Number of causal markers", "AUC", "FDR", "TPR"))
print("-"*62)
print("| %3d | %25d | %6.4f | %6.4f | %6.4f |" % (it, sum(pvals <= pval_th), area, fdr, tpr))
print("-"*62)
print("\n", flush=True)