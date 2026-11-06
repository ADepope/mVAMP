# mVAMP
mVAMP is Vector Approximate Message Passing method for analyzing omics data, written in C++ (source code in src/).

This repository also contains scripts for data simulations and post-processing on methylation data.

simulation/sim_top_iid.py - loading adjusted and standardized methylation data from zarr files and simulate iid phenotype on the top.

# Example
## Data simulation
Python script ``data_sim.py`` can be used to generate example i.i.d. design matrix (``example.bin``), marker effects (``example_ts.bin``) and coresponding continues outcome (``example.phen``)
```
python3 data_sim.py --out-dir ...
```

This creates following files in specified output directory:
 - ``example.bin`` file with data for each sample. Data are stored as 8 byte DOUBLE precision numbers. Whole file is sequence of M marker blocks, where each block contains N 8 byte DOUBLE precision values corespoding to each sample.
 - ``example_ts.bin`` file with true signals in binary format. File contains M 8 byte DOUBLE precision values, coresponding to true signals used for simulation.
 - ``example.phen`` file with simulated phenotype in PLINK format.

## Compilation
Assuming standard HPC environment, we can compile the software using mpic++. Path to source code has to be specified.

```
module load gcc openmpi boost

vloc={Path to mVAMP/src folder}

mpic++ ${vloc}/main_meth.cpp ${vloc}/vamp.cpp ${vloc}/utilities.cpp ${vloc}/data.cpp ${vloc}/options.cpp -march=native -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  ${vloc}/main_meth.exe
```

## Running mVAMP
### Inference
Minimal setup of input options that has to be specified for inference mode is following:

```
export OMP_NUM_THREADS={Number of OpenMP threads}

mpirun -np {Number of MPI ranks} ${vloc}/main_meth.exe \                      
                                --meth-file example.bin \
                                --phen-file example.phen \
                                --N 1000 \
                                --Mt 2000 \
                                --out-dir output/ \
                                --out-name example \
```

Output files:

- ``example_params.csv`` table containing the model's hyper-parameters over iterations. Column names: iteration, alpha1, gam1, alpha2, gam2, gamw.

- ``example_metrics.csv`` table containing performace metrics over iterations. Column names: iteration, R2_denoising, Corr(xhat1,x0), R2_lmmse, Corr(xhat2,x0), Corr(yhat1,y)^2, Corr(yhat2,y)^2.

- ``example_prior.csv`` table containing the prior parameters over iterations. Column names: iteration, number of mix. components L, L * prior probabilities, L * prior variances

- ``example_it_{iteration}.bin`` binary files containing signal estimates from denoising step xhat1 in current iteration. 8 byte DOUBLE precision values.

- ``example_r1_it_{iteration}.bin`` binary files containing noise corrupted signal estimates from denoising step r1 in current iteration. 8 byte DOUBLE precision values.

### Testing
If test subset is available, running mVAMP for out-of-sample testing requires at least:

```
export OMP_NUM_THREADS={Number of OpenMP threads}

mpirun -np {Number of MPI ranks} ${vloc}/main_meth.exe \                      
                                --meth-file-test example.bin \
                                --phen-file-test example.phen \
                                --N-test 1000 \
                                --Mt 2000 \
                                --out-dir output/ \
                                --out-name example \
                                --estimate-file example_it_1.bin \
                                --run-mode test \
```
Output file:

- ``example_test.csv`` table containing out-of-sample metrics over iterations. Column names: iteration, R2 test, Corr(yhat,y)^2

### Association testing
mVAMP framework allows to test statistical significance of marker associations. Currently supported pval methods (--pval-method) are state evolution (se) or leave one out (loo) testing. State evolution p value testing example: 
```
export OMP_NUM_THREADS={Number of OpenMP threads}

mpirun -np {Number of MPI ranks} ${vloc}/main_meth.exe \                      
                                --meth-file-test example.bin \
                                --phen-file-test example.phen \
                                --N 1000 \
                                --Mt 2000 \
                                --out-dir output/ \
                                --out-name example \
                                --r1-file example_r1_it_{iteration}.bin \
                                --gam1 {gam1 value from iteration} \
                                --run-mode association_test \
                                --pval-method se \
```

Output file:
- ``example_it_{iteration}_pval_se.bin`` binary file containing p values from state evlolution. 8 byte DOUBLE precision values.

Leave one out (loo) p value testing example: 
```
export OMP_NUM_THREADS={Number of OpenMP threads}

mpirun -np {Number of MPI ranks} ${vloc}/main_meth.exe \                      
                                --meth-file-test example.bin \
                                --phen-file-test example.phen \
                                --N 1000 \
                                --Mt 2000 \
                                --out-dir output/ \
                                --out-name example \
                                --estimate-file example_it_{iteration}.bin \
                                --run-mode association_test \
                                --pval-method loo \
```

Output file:
- ``example_it_{iteration}_pval_loo.bin`` binary file containing p values from leave one out testing. 8 byte DOUBLE precision values.

Eventully, state evolution p value testing can be done using python script ``scripts/p_vals.py``. Example:
```
python3 p_vals.py   --out-name example \
                    --csv-params example_params.csv \
                    --r1-file example_r1_it_{iteration}.bin \
                    --it {iteration} \
                    --th 0.05 \
                    --M 2000 \
                    --N 1000 \
```
Output file:
- ``example_it_{iteration}_pval.bin`` binary file containing p values for each marker position in current iteration. 8 byte DOUBLE precision values.

# Input options

| Option | Description | Default |
| --- | --- | --- |
| `--run-mode` | `infere` / `test` / `association_test` | `infere` |
| `--model` | Regression model `linear` / `bin_class` | `linear` |
| `--pval_method` | Method used for calculating P values in association testing `se` (State Evolution) / `loo` (Leave-One-Out) | `se` |
| `--meth-file` | Path to .bin file including the data in binary format | |
| `--meth-file-test` | Path to .bin file for testing | |
| `--phen-file` | Path to file containing phenotype data | |
| `--phen-file-test` | Path to phenotype file for testing | |
| `--cov-file` | Path to .cov file including covariates | |
| `--cov-file-test` | Path to .cov file including covariates for testing | |
| `--estimate-file` | Path to file including signal estimates for testing | |
| `--r1-file` | Path to file including r1 vector - noised version of signal estimates | |
| `--N` | Number of individuals for inference | |
| `--N-test` | Number of individuals for testing | |
| `--Mt` | Total number of markers for inference | |
| `--C` | Indicates the number of covariates | |
| `--out-dir` | Output directory for the signal estimates | |
| `--out-name` | Name of the output file | |
| `--iterations` | Maximal number of iterations | 10 |
| `--test-iter-range` | Iteration range for testing | 1,10 |
| `--rho` | Damping factor | 0.5 |
| `--gam1` | Signal noise precision. Used for initialization when ``infer`` mode, used for p value calculation when ``association_test`` mode | 1e-6 |
| `--h2` | Heritability value used in simulations | 0.5 |
| `--true_signal_file` | Path to binary file containing true signals | |
| `--num-mix-comp` | Nnumber of gaussian mixture components | 2 |
| `--probs` | Initial prior mixture coefficients (separated by comma, must sum up to 1) | 0.5,0.5 |
| `--vars` | Initial prior variances (separated by comma) | 0.0,0.001|
| `--learn-vars` | Indicates whether or not prior variances are learned | 0 |
| `--CG-max-iter` | Maximal number of iteration used in conjugate gradient method | 500 |
| `--EM-err-thr` | Relative error threshold within expectation maximization | 1e-2 |
| `--EM-max-iter` | Maximal number of iterations of expectation maximization | 1 |
| `--stop-criteria-thr` | Relative error threshold within expectation maximization | 0 |
| `--verbosity` | Whether or not to print out details | 0 |


# Initialization from Gibbs sampler
Efficient implementation of Gibbs sampler for multi-omics data, GMRMomi, can be used for initializing mVAMP algorithm using very first samples from MCMC posterior trace.

Example:
1. Run GMRMomi, as in https://github.com/medical-genomics-group/gmrm-omi
2. Run ``conf_gibbs_init.py`` to create .conf configuration file for mVAMP initialization.
```
python3 conf_gibbs_init.py  --csv example.csv \
                            --grm example.grm \
                            --out-dir output/directory/ \
                            --iterations 100:200 \
                            --rho 0.5
```
3. Access the configuration and run mVAMP
```
# If more than one configuration in the file, use different IDs
id=0

# Configuration file path
conf=example.conf

# Accessing the values in .conf file
rho=$(awk -v ID=$id '$1==ID {print $2}' $conf)
L=$(awk -v ID=$id '$1==ID {print $3}' $conf)
probs=$(awk -v ID=$id '$1==ID {print $5}' $conf)
vars=$(awk -v ID=$id '$1==ID {print $6}' $conf)
h2=$(awk -v ID=$id '$1==ID {print $7}' $conf)

# Running mVAMP
export OMP_NUM_THREADS={Number of OpenMP threads}

mpirun -np {Number of MPI ranks} ${vloc}/main_meth.exe \                      
                                --meth-file example.bin \
                                --phen-file example.phen \
                                --N 1000 \
                                --Mt 2000 \
                                --out-dir output/ \
                                --out-name example \
                                --num-mix-comp $L \
                                --probs $probs \
                                --vars $vars \
                                --rho $rho \
                                --h2 $h2
```