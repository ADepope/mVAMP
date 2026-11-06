#pragma once
#include <vector>
#include <random>
#include <string>
#include <mpi.h>
#include "data.hpp"
#include "options.hpp"
#include "utilities.hpp"
class vamp{

private:
    int N, M, Mt, C, max_iter, rank, nranks;
    double gam1, gam_before;
    double gam2 = 0;
    double eta1 = 0;
    double eta2 = 0;
    double tau1, tau2;                          // probit model precisions
    double alpha1 = 0;                          // Onsager corrections
    double alpha2;
    double rho;                                 // damping factor
    double gamw;                                // linear model noise precision

    std::vector<double> x1_hat, x2_hat, true_signal, x1hat_init;
    std::vector<double> z1_hat, z2_hat;
    std::vector<double> y;                              // phenotype vector
    std::vector<double> z1;                             // z1 = A * x1_hat
    std::vector<double> r1, r2, r2_prev;
    std::vector<double> p1, p2;
    std::vector<double> cov_eff;                        // covariates in a probit model
    std::vector<double> mu_CG_last;                     // last LMMSE estimate

    std::vector<double> probs, probs_before;
    std::vector<double> vars, vars_before;

    double gamma_min = 1e-11;
    double gamma_max = 1e11;
    double probit_var = 1;
    int EM_max_iter = 100;
    double EM_err_thr = 1e-2;
    int CG_max_iter = 100;
    double CG_err_tol = 1e-5;
    int learn_vars = 1;
    int learn_prior_delay = 1;
    double damp_max = 1;
    double damp_min = 0.05;
    double stop_criteria_thr = 0.01;
    double merge_vars_thr = 5e-1;

    std::string model;
    std::string out_dir;
    std::string out_name;

    std::random_device rd; // random device for sampling
    std::vector<double> bern_vec;
    std::vector<double> invQ_bern_vec;

    double total_comp_time = 0;

    int verbosity = 0;

    MPI_File out_params_fh;
    MPI_File out_metrics_fh;
    MPI_File out_prior_fh;

    // Headers for output files
    std::vector<std::string> metrics_header{"iteration",
                                            "R2 denoising",
                                            "x1 correlation denoising",
                                            "R2 LMMSE",
                                            "x2 correlation LMMSE",
                                            "z1 correlation denoising",
                                            "z2 correlation LMMSE"};

    std::vector<std::string> params_header{"iteration",
                                            "alpha1",
                                            "gam1",
                                            "alpha2",
                                            "gam2",
                                            "gamw"};

    std::vector<std::string> prior_header{"iteration", "number of components"}; // rest of the header is specified in io_setup();

public:
    // ---------- CONSTRUCTOR -----------
    vamp( int N,
            int M,
            int Mt,
            int C,
            double gam1,
            double gamw,
            int max_iter,
            int CG_max_iter,
            double CG_err_tol,
            int EM_max_iter,
            double EM_err_thr,
            double rho,
            int learn_vars,
            int learn_prior_delay,
            double stop_criteria_thr,
            double merge_vars_thr,
            std::vector<double> vars,
            std::vector<double> probs,
            std::vector<double> true_signal,
            std::vector<double> x1hat_init,
            std::string out_dir,
            std::string out_name,
            std::string model,
            int verbosity,
            int rank);

    //  INFERENCE
    std::vector<double> infere(data* dataset);
    std::vector<double> infere_linear(data* dataset);
    std::vector<double> infere_bin_class(data* dataset);

    // DENOISING PROCEDURES & ONSAGER CALCULATION
    double g1(double x, double gam1);
    double g1_bin_class(double p, double tau1, double y, double m_cov);
    double g1d(double x, double gam1);
    double g1d_bin_class(double p, double tau1, double y, double m_cov);
    double g2d_onsager(double gam2, double tau, data* dataset);

    // HYPERPARAMETERS UPDATE
    void updatePrior();
    void updateNoisePrec(data* dataset);
    void updateNoisePrecAAT(data* dataset);

    std::vector<double> lmmse_mult(std::vector<double> v, double tau, data* dataset);

    std::vector<double> precondCG_solver(std::vector<double> v, double tau, int denoiser, data* dataset);
    std::vector<double> precondCG_solver(std::vector<double> v, std::vector<double> mu_start, double tau, int denoiser, data* dataset);

    void err_measures(data * dataset, int ind, std::vector<double>& metrics);
    void probit_err_measures(data *dataset, int sync, std::vector<double> true_signal, std::vector<double> est, std::string var_name);
    std::vector<int> confusion_matrix(std::vector<double> y, std::vector<double> yhat);

    double probit_var_EM_deriv(double v, std::vector<double> z, std::vector<double> y);
    double expe_probit_var_EM_deriv(double v, double eta, std::vector<double> z, std::vector<double> y);
    double update_probit_var(double v, double eta, std::vector<double> z_hat, std::vector<double> y);
    std::vector<double> grad_cov(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta);
    double mlogL_probit(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta);
    std::vector<double> grad_desc_step_cov(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta, double* grad_norm);
    std::vector<double> grad_desc_cov(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta);
    std::vector<double> Newton_method_cov(std::vector<double> y, std::vector<double> gg, std::vector< std::vector<double> > Z, std::vector<double> eta);

    void set_gam2 (double gam) { gam2 = gam; }
    std::vector<double> get_cov_eff() const {return cov_eff;}
    std::vector<double> predict_probit(std::vector<double> z, double th);
    double accuracy(std::vector<int> conf_mat);

    void setup_io();

};
