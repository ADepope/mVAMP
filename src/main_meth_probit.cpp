#include <mpi.h>
#include <iostream>
#include <numeric>
#include "utilities.hpp"
#include "data.hpp"
#include "vamp.hpp"

int main(int argc, char** argv)
{
    // starting parallel MPI processes
    int required_MPI_level = MPI_THREAD_MULTIPLE;
    int provided_MPI_level;
    int rank = 0; // current rank number
    int nranks = 0; //number of ranks
    MPI_Init_thread(NULL, NULL, required_MPI_level, &provided_MPI_level);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Command line options
    const Options opt(argc, argv);

    std::string out_dir = opt.get_out_dir();
    std::string out_name = opt.get_out_name();
    std::string model = opt.get_model();
    int verbosity = opt.get_verbosity();

    size_t Mt = opt.get_Mt(); // Total number of markers
    size_t N = opt.get_N(); // Number of samples
    int C = opt.get_C(); // Number of covariates

    // Divide work for parallel ranks
    std::vector<double> MS = divide_work(Mt);
    int M = MS[0]; // Number of markers for current rank
    int S = MS[1]; // Offset of part of markers processed by current rank

    // Infer model parameters and calculate test R2
    if (opt.get_run_mode() == "infere"){

        // Get command line options
        std::string phenfp = opt.get_phen_file(); // Phenotype file
        std::string mrkfp = opt.get_meth_file();
        std::string covfp = opt.get_cov_file();
        double alpha_scale = opt.get_alpha_scale();

        // Reading train set
        data dataset(phenfp, mrkfp, model, N , M, Mt, S, rank, alpha_scale);

        if(C > 0)
            dataset.read_covariates(covfp, C);

        // Initialize model hyperparameters
        double h2 = opt.get_h2(); // heritability
        double gamw = 1.0 / (1.0 - h2); // Noise precision
        double gam1 = opt.get_gam1(); // Signal noise precision

        int max_iter = opt.get_iterations();
        int learn_vars = opt.get_learn_vars();
        int learn_prior_delay = opt.get_learn_prior_delay();
        double stop_criteria_thr = opt.get_stop_criteria_thr();
        double merge_vars_thr = opt.get_merge_vars_thr();
        int CG_max_iter = opt.get_CG_max_iter();
        double CG_err_tol = opt.get_CG_err_tol();
        int EM_max_iter = opt.get_EM_max_iter();
        double EM_err_thr = opt.get_EM_err_thr();
        double rho = opt.get_rho();
        std::vector<double> vars = opt.get_vars();
        std::vector<double> probs = opt.get_probs();
        std::vector<double> x1hat_init = std::vector<double> (M, 0.0);

        // Read true signals, if provided
        std::vector<double> true_signal;
        if(opt.get_true_signal_file() != "")
            true_signal = mpi_read_vec_from_file(opt.get_true_signal_file(), M, S);
        else
            true_signal = std::vector<double> (M, 0.0);

        // ---------------- running VAMP algorithm -------------------- //
        vamp emvamp(N,
                    M,
                    Mt,
                    C,
                    gam1,
                    gamw,
                    max_iter,
                    CG_max_iter,
                    CG_err_tol,
                    EM_max_iter,
                    EM_err_thr,
                    rho,
                    learn_vars,
                    learn_prior_delay,
                    stop_criteria_thr,
                    merge_vars_thr,
                    vars,
                    probs,
                    true_signal,
                    x1hat_init,
                    out_dir,
                    out_name,
                    model,
                    verbosity,
                    rank); // Create VAMP instance

        std::vector<double> x_est = emvamp.infere(&dataset); // Infer model parameters

    }
    else if (opt.get_run_mode() == "test") // just analyzing the result on the test data
    {
        // Setup out csv file
        std::string outcsv_fp = out_dir + "/" + out_name + "_test.csv";
        MPI_File_delete(outcsv_fp.c_str(), MPI_INFO_NULL);
        MPI_File outcsv_fh;
        check_mpi(  MPI_File_open(MPI_COMM_WORLD,
                outcsv_fp.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL,
                MPI_INFO_NULL,
                &outcsv_fh),
                __LINE__, __FILE__);

        // reading test set
        const std::string mrkfp_test = opt.get_meth_file_test();
        const std::string pheno_test = opt.get_phen_file_test(); // currently it is only supported passing one pheno files as an input argument

        size_t N_test = opt.get_N_test();

        double alpha_scale = opt.get_alpha_scale();

        data dataset_test(pheno_test, mrkfp_test, model, N_test, M, Mt, S, rank, alpha_scale);

        std::vector<double> y = dataset_test.get_phen();

        // parse estimate file name
        std::string est_file_name = opt.get_estimate_file();
        int pos_dot = est_file_name.find(".");
        std::string end_est_file_name = est_file_name.substr(pos_dot + 1);
        if (rank == 0)
            std::cout << "est_file_name = " << est_file_name << std::endl;

        int pos_it = est_file_name.rfind("it");
        std::vector<int> iter_range = opt.get_test_iter_range();
        int min_it = iter_range[0];
        int max_it = iter_range[1];
        if (rank == 0)
            std::cout << "iter range = [" << min_it << ", " << max_it << "]" << std::endl;

        for (int it = min_it; it <= max_it; it++){

            std::vector<double> params(5,0); // Parameters to save to csv [TP, TN, FP, FN, ACC]

            std::vector<double> x_est;
            std::string est_file_name_it = est_file_name.substr(0, pos_it) + "it_" + std::to_string(it) + "." + end_est_file_name;

            if (end_est_file_name == "bin")
                x_est = mpi_read_vec_from_file(est_file_name_it, M, S);
            else
                x_est = read_vec_from_file(est_file_name_it, M, S);

            // normalization of estimates
            for (int i0 = 0; i0 < x_est.size(); i0++)
                x_est[i0] *= sqrt( (double) N_test );

            // Predictions
            std::vector<double> z_test = dataset_test.Ax(x_est.data());
            std::vector<double> yhat(N_test,0);
            for (int i=0; i < N_test; i++){
                double prob = normal_cdf(z_test[i]);
                if (prob >= 0.5)
                    yhat[i] = 1;
                else
                    yhat[i] = 0;
            }

            int TP=0, TN=0, FP=0, FN=0;
            for (int i = 0; i < N_test; i++){
                if (y[i] == 1 && yhat[i] == 1)
                    TP++;
                else if (y[i] == 0 && yhat[i] == 0)
                    TN++;
                else if (y[i] == 1 && yhat[i] == 0)
                    FN++;
                else if (y[i] == 0 && yhat[i] == 1)
                    FP++;
            }
            double ACC = (double)(TP + TN ) / (double)(TP + TN + FP + FN);

            params[0] = TP;
            params[1] = TN;
            params[2] = FP;
            params[3] = FN;
            params[4] = ACC;

            if (rank == 0){
                std::cout << "---- Iteration " << it << "----" << std::endl;
                std::cout <<  "TP = " << TP <<  std::endl;
                std::cout <<  "TN = " << TN <<  std::endl;
                std::cout <<  "FP = " << FP <<  std::endl;
                std::cout <<  "FN = " << FN <<  std::endl;
                std::cout <<  "Accuracy = " << ACC <<  std::endl;
                write_ofile_csv(outcsv_fh, it, &params);
            }
        }
    }
    else if (opt.get_run_mode() == "predict") // store linear predictor A * x_est for the test data
    {
        // reading test set
        const std::string mrkfp_test = opt.get_meth_file_test();
        const std::string pheno_test = opt.get_phen_file_test(); // currently it is only supported passing one pheno files as an input argument
        std::string est_file_name = opt.get_estimate_file();
        int pos_it = est_file_name.rfind("it");
        std::string pred_file_name = est_file_name.substr(0, pos_it) + ".yhat";

        size_t N_test = opt.get_N_test();
        double alpha_scale = opt.get_alpha_scale();

        data dataset_test(pheno_test, mrkfp_test, model, N_test, M, Mt, S, rank, alpha_scale);

        std::vector<double> x_est = mpi_read_vec_from_file(est_file_name, M, S);

        // normalization of estimates
        for (int i0 = 0; i0 < x_est.size(); i0++)
            x_est[i0] *= sqrt( (double) N_test );

        // Predictions
        std::vector<double> z_hat = dataset_test.Ax(x_est.data());

        store_vec_to_file(pred_file_name, z_hat);
    }
    else if (rank == 0) {
        std::cout << "FATAL: unknown --run-mode \"" << opt.get_run_mode() << "\" (use infere, test or predict)" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
