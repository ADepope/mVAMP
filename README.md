# mVAMP

mVAMP fits sparse Bayesian regression models of an outcome on DNA methylation data using
**Vector Approximate Message Passing (VAMP)**. It is written in C++ and parallelised with MPI
(markers are split across ranks) and OpenMP (threads within a rank).

What it does:

- **Linear model** for continuous outcomes, and a **probit model** for binary outcomes.
- **Adaptive prior**: marker effects follow a spike-and-slab mixture of Gaussians. The mixture
  probabilities and variances are learned during inference with expectation maximisation (EM).
- **Out-of-sample testing** of the estimated effects on a held-out set.
- **Association testing**: a p-value for each marker, from state evolution (`se`) or
  leave-one-out regression (`loo`).

## Repository layout

```
src/          C++ source: VAMP inference (vamp.cpp, vamp_probit.cpp), data handling, options
simulation/   Python scripts to simulate example data
scripts/      Python scripts to post-process mVAMP output (p-values, plots, metrics)
```

## Requirements

- A C++17 compiler with OpenMP support (tested with GCC)
- An MPI implementation (e.g. OpenMPI)
- Boost (`math` and `numeric/ublas`, header-only)
- For the Python scripts: `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, and `zarr`
  for `simulation/sim_top_iid.py`

## Compilation

There are two executables: `main_meth.exe` for the linear model and `main_meth_probit.exe` for
the probit model. Both are compiled together with `vamp.cpp`, which also includes
`vamp_probit.cpp`.

```bash
module load gcc openmpi boost   # or however your system provides them

vloc=path/to/mVAMP/src
FLAGS="-march=native -Ofast -fopenmp -lstdc++fs"

mpic++ ${vloc}/main_meth.cpp ${vloc}/vamp.cpp ${vloc}/utilities.cpp ${vloc}/data.cpp ${vloc}/options.cpp ${FLAGS} -o ${vloc}/main_meth.exe

mpic++ ${vloc}/main_meth_probit.cpp ${vloc}/vamp.cpp ${vloc}/utilities.cpp ${vloc}/data.cpp ${vloc}/options.cpp ${FLAGS} -o ${vloc}/main_meth_probit.exe
```

For debugging, add `-g -D_GLIBCXX_DEBUG`. This turns on bounds-checked standard containers,
which makes runs much slower.

## Input data

| File | Format |
| --- | --- |
| Methylation data (`--meth-file`) | Binary file of 8-byte doubles, stored marker by marker: `M` blocks, each with the `N` values of one marker (one per individual). Markers are standardised internally. |
| Outcome (`--phen-file`) | Text file in PLINK phenotype format, one row per individual, no header: `FID IID value`. No missing values (`NA`) are allowed. For the linear model the outcome is scaled to unit variance but **not centred**, so centre it beforehand. For the probit model use values `0`/`1`. |
| Covariates (`--cov-file`, probit model only) | Space-separated text file with one header row, then one row per individual: `FID IID cov_1 ... cov_C`. Covariates are standardised internally. |
| True effects (`--true-signal-file`, optional) | Binary file of `M` 8-byte doubles. Only used to report accuracy metrics in simulations. |

## Example

### 1. Simulate data

```bash
python3 simulation/data_sim.py --out-dir data/ --N 1000 --M 2000
```

This writes `data/example.bin` (methylation matrix), `data/example.phen` (outcome) and
`data/example_ts.bin` (true effects).

To simulate an outcome on top of real methylation data stored as one zarr file per chromosome,
with a train/test split, use `simulation/sim_top_iid.py`.

### 2. Inference

```bash
export OMP_NUM_THREADS={number of OpenMP threads}

mpirun -np {number of MPI ranks} ${vloc}/main_meth.exe \
    --meth-file data/example.bin \
    --phen-file data/example.phen \
    --N 1000 \
    --Mt 2000 \
    --out-dir output/ \
    --out-name example
```

Add `--true-signal-file data/example_ts.bin` to also get the correlation between the estimated and
true effects in `example_metrics.csv`.

Output files:

| File | Content |
| --- | --- |
| `example_params.csv` | Hyperparameters per iteration: `iteration, alpha1, gam1, alpha2, gam2, gamw` |
| `example_metrics.csv` | Metrics per iteration: `iteration`, R² and correlation with the true effects for the denoising step, the same for the LMMSE step, then squared correlation between predicted and observed outcome for both steps |
| `example_prior.csv` | Learned mixture probabilities and variances. Rows are written only by the probit model; the linear model prints them to standard output instead. |
| `example_it_{k}.bin` | Effect estimates (`x1_hat`, denoising step) after iteration `k`, `M` 8-byte doubles |
| `example_r1_it_{k}.bin` | Noisy effect estimates `r1` after iteration `k`, used for state evolution p-values |

The CSV files are written by MPI at fixed offsets and can contain NUL padding bytes. The scripts in
`scripts/` strip these when reading.

### 3. Out-of-sample testing

Evaluates the estimates from a range of iterations on a test set. Pass any one of the
`_it_{k}.bin` files as `--estimate-file`; the iteration number in its name is replaced by each
value in `--test-iter-range`. The path must contain no other `.` than the one before the
extension (so use `output/`, not `./output/`). The test files can come from, for example, the
train/test split made by `simulation/sim_top_iid.py`.

```bash
mpirun -np {number of MPI ranks} ${vloc}/main_meth.exe \
    --run-mode test \
    --meth-file-test data/example_test.bin \
    --phen-file-test data/example_test.phen \
    --N-test 1000 \
    --Mt 2000 \
    --estimate-file output/example_it_1.bin \
    --test-iter-range 1,10 \
    --out-dir output/ \
    --out-name example
```

Output: `example_test.csv` with `iteration`, test R² and squared correlation between predicted and
observed outcome.

### 4. Association testing

State evolution p-values, from the `r1` vector of iteration `k` and that iteration's `gam1` (from
`example_params.csv`):

```bash
mpirun -np {number of MPI ranks} ${vloc}/main_meth.exe \
    --run-mode association_test \
    --pval-method se \
    --meth-file data/example.bin \
    --phen-file data/example.phen \
    --N 1000 \
    --Mt 2000 \
    --r1-file output/example_r1_it_{k}.bin \
    --gam1 {gam1 from iteration k} \
    --out-dir output/ \
    --out-name example
```

Output: `example_it_{k}_pval_se.bin`, `M` 8-byte doubles.

Leave-one-out p-values, from the effect estimates of iteration `k`:

```bash
mpirun -np {number of MPI ranks} ${vloc}/main_meth.exe \
    --run-mode association_test \
    --pval-method loo \
    --meth-file data/example.bin \
    --phen-file data/example.phen \
    --N 1000 \
    --Mt 2000 \
    --estimate-file output/example_it_{k}.bin \
    --out-dir output/ \
    --out-name example
```

Output: `example_it_{k}_pval_loo.bin`, `M` 8-byte doubles.

### Binary outcomes (probit model)

Use `main_meth_probit.exe` with `--model bin_class` and a 0/1 outcome. Covariates can be given with
`--cov-file` and `--C`. Run modes:

- `infere`: inference, with the same output files as above. The metrics file holds the confusion
  matrix (TP, TN, FP, FN), accuracy and correlation with the true effects, for both steps.
- `test`: confusion matrix and accuracy on a test set for a range of iterations, written to
  `{out-name}_test.csv`.
- `predict`: writes the linear predictor for the test set, computed from `--estimate-file`, to a
  `.yhat` text file next to the estimate file.

## Command-line options

| Option | Description | Default |
| --- | --- | --- |
| `--run-mode` | `infere`, `test` or `association_test` (`main_meth.exe`); `infere`, `test` or `predict` (`main_meth_probit.exe`) | `infere` |
| `--model` | `linear` or `bin_class` (probit) | `linear` |
| `--meth-file` | Methylation data for inference | |
| `--phen-file` | Outcome for inference | |
| `--meth-file-test` | Methylation data for testing | |
| `--phen-file-test` | Outcome for testing | |
| `--cov-file` | Covariates (probit model only) | |
| `--C` | Number of covariates | 0 |
| `--N` | Number of individuals for inference | |
| `--N-test` | Number of individuals for testing | |
| `--Mt` | Total number of markers | |
| `--out-dir` | Output directory | |
| `--out-name` | Prefix for output files | |
| `--true-signal-file` | True effects, for simulations | |
| `--estimate-file` | Effect estimates for testing and leave-one-out p-values | |
| `--r1-file` | `r1` vector for state evolution p-values | |
| `--pval-method` | `se` (state evolution) or `loo` (leave-one-out) | `se` |
| `--iterations` | Maximum number of VAMP iterations | 50 |
| `--test-iter-range` | First and last iteration to evaluate in `test` mode | `1,50` |
| `--stop-criteria-thr` | Stop when the relative change of the estimates falls below this | 0.01 |
| `--rho` | Damping factor | 0.5 |
| `--h2` | Initial heritability; sets the initial noise precision to `1 / (1 - h2)` | 0.5 |
| `--gam1` | Initial signal noise precision; in `association_test` mode with `se`, the `gam1` of the chosen iteration | 1e-6 |
| `--probs` | Initial mixture probabilities, comma-separated, summing to 1. The first component is the spike at zero. | 10 components |
| `--vars` | Initial mixture variances, comma-separated, same length as `--probs`. The first should be 0. | 10 components |
| `--learn-vars` | Learn the mixture variances (1) or keep them fixed (0) | 1 |
| `--learn-prior-delay` | Start learning the prior after this many iterations | 1 |
| `--EM-max-iter` | Maximum EM iterations per prior update | 1 |
| `--EM-err-thr` | Relative error threshold for EM | 0.01 |
| `--merge-vars-thr` | Merge mixture components whose variances differ by less than this relative amount | 0.5 |
| `--CG-max-iter` | Maximum conjugate gradient iterations | 500 |
| `--CG-err-tol` | Relative error tolerance for conjugate gradient | 1e-5 |
| `--alpha-scale` | Markers are scaled by `1 / sd^alpha-scale` | 1 |
| `--verbosity` | Print details of the EM and CG iterations (1) | 0 |

The defaults for `--probs` and `--vars` are a 10-component mixture with variances
`0, 1e-6, 6e-6, 3e-5, 2e-4, 1e-3, 6e-3, 3e-2, 0.2, 1`. The spike gets probability 0.99, and the
remaining 0.01 is spread over the other components, halving from one component to the next.

## Post-processing scripts

| Script | Purpose |
| --- | --- |
| `scripts/p_vals.py` | State evolution p-values computed in Python from an `r1` file and `_params.csv` (same as `--pval-method se`) |
| `scripts/metrics.py` | Plots R² (train and test), `gamw` and `gam1` over iterations, and prints a summary table |
| `scripts/manhattan.py` | Manhattan plot of p-values; needs per-chromosome probe lists |
| `scripts/roc.py` | ROC curve, AUC, FDR and TPR of p-values against known true effects (simulations) |
| `scripts/r2.py` | R² between predicted and observed outcome |

Example:

```bash
python3 scripts/p_vals.py --out-name example_it_{k}_pval \
                          --csv-params output/example_params.csv \
                          --r1-file output/example_r1_it_{k}.bin \
                          --it {k} \
                          --M 2000 \
                          --N 1000
```

This writes `output/example_it_{k}_pval.bin` and prints how many markers pass the Bonferroni
threshold `--th / M` (default `--th 0.05`).
