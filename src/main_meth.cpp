#include <mpi.h>
#include <iostream>
#include <numeric>
#include "utilities.hpp"
#include "data.hpp"
#include "vamp.hpp"
#include <boost/math/distributions/normal.hpp>

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
        std::string est_file_name = opt.get_estimate_file();
        double alpha_scale = opt.get_alpha_scale();

        // Reading train set
        data dataset(phenfp, mrkfp, model, N , M, Mt, S, rank, alpha_scale);

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

        // Read true signals, if provided
        std::vector<double> true_signal;
        if(opt.get_true_signal_file() != "")
            true_signal = mpi_read_vec_from_file(opt.get_true_signal_file(), M, S);
        else
            true_signal = std::vector<double> (M, 0.0);

        // Read x1hat_init, if provided
        std::vector<double> x1hat_init;
        if(est_file_name != "")
            x1hat_init = mpi_read_vec_from_file(est_file_name, M, S);
        else
            x1hat_init = std::vector<double> (M, 0.0);

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

        // reading test set
        const std::string mrkfp_test = opt.get_meth_file_test();
        const std::string pheno_test = opt.get_phen_file_test(); // currently it is only supported passing one pheno files as an input argument

        int N_test = opt.get_N_test();
        double alpha_scale = opt.get_alpha_scale();

        data dataset_test(pheno_test, mrkfp_test, model, N_test, M, Mt, S, rank, alpha_scale);

        std::vector<double> y_test = dataset_test.get_phen();

        // Open output file for storing test metrics (R2)
        MPI_File outcsv_test_fh;
        std::string outcsv_test_fp = opt.get_out_dir() + "/" + opt.get_out_name() + "_test.csv";
        MPI_File_delete(outcsv_test_fp.c_str(), MPI_INFO_NULL);
        check_mpi(MPI_File_open(MPI_COMM_WORLD,
                            outcsv_test_fp.c_str(),
                            MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL,
                            MPI_INFO_NULL,
                            &outcsv_test_fh),
                            __LINE__, __FILE__);
        // Headers for output files
        std::vector<std::string> test_header{"iteration",
                                            "R2 test",
                                            "z correlation test"};
        if(rank == 0)
            write_ofile_csv_header(outcsv_test_fh, &test_header);

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
            std::vector<double> params;
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

            // L2 prediction error
            double l2_pred_err2 = 0;
            for (int i0 = 0; i0 < N_test; i0++){
                l2_pred_err2 += (y_test[i0] - z_test[i0]) * (y_test[i0] - z_test[i0]);
            }

            // Standard deviation of true outcome
            double stdev = calc_stdev(y_test);
            double r2 = 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ); // R2 test

            // Correlation squared
            double corr_y = inner_prod(z_test, y_test, 1) / sqrt( l2_norm2(z_test, 1) * l2_norm2(y_test, 1) );
            double corr_y_2 = corr_y * corr_y;

            if (rank == 0){
                std::cout << r2 << ", ";
                params.push_back(r2);
                params.push_back(corr_y_2);
                write_ofile_csv(outcsv_test_fh, it, &params);
            }
        }
    }
    else if (opt.get_run_mode() == "association_test"){

        // Get command line options
        std::string phenfp = opt.get_phen_file(); // Phenotype file
        std::string mrkfp = opt.get_meth_file();
        double alpha_scale = opt.get_alpha_scale();

        // Reading data set
        data dataset(phenfp, mrkfp, model, N , M, Mt, S, rank, alpha_scale);

        std::string pval_method = opt.get_pval_method();
        std::string filepath_out;

        if(pval_method == "se"){ // TODO: SE option not tested
            // Read r vector
            std::string r1_file_name = opt.get_r1_file();
            int pos1 = r1_file_name.rfind("it_") + 3;
            int pos2 = r1_file_name.rfind(".bin");
            std::string it_str = r1_file_name.substr(pos1, pos2 - pos1);
            if (rank == 0)
                std::cout << r1_file_name << std::endl;
            std::vector<double> r1 = mpi_read_vec_from_file(r1_file_name, M, S);

            double gam1 = opt.get_gam1(); // Signal noise precision

            std::vector<double> pvals(M, 0.0);
            for(int j=0; j < M; j++){
                boost::math::normal norm(r1[j], sqrt(1.0 / (gam1 * (double) N )));
                double pval = boost::math::cdf(norm, 0);
                if(r1[j] <= 0.0)
                    pval = 1 - pval;
                pvals[j] = pval;
            }
            filepath_out = out_dir + "/" + out_name + "_it_" + it_str + "_pval_se.bin";
            if (rank == 0)
                std::cout << "Storing p-values to file " + filepath_out << std::endl;
            mpi_store_vec_to_file(filepath_out, pvals, S, M);
        }
        else if(pval_method == "loo"){
            // parse estimate file name
            std::string est_file_name = opt.get_estimate_file();
            int pos1 = est_file_name.rfind("it_") + 3;
            int pos2 = est_file_name.rfind(".bin");
            std::string it_str = est_file_name.substr(pos1, pos2 - pos1);
            std::vector<double> x1_hat = mpi_read_vec_from_file(est_file_name, M, S);
            // normalization of estimates
            for (int i0 = 0; i0 < x1_hat.size(); i0++)
                x1_hat[i0] *= sqrt( (double) N );

            std::vector<double> z1 = dataset.Ax(x1_hat.data());
            std::vector<double> y = dataset.get_phen();
            std::vector<double> pvals = dataset.pvals_loo(z1, y, x1_hat);
            filepath_out = out_dir + "/" + out_name + "_it_" + it_str + "_pval_loo.bin";
            if (rank == 0)
                std::cout << "Storing p-values to file " + filepath_out << std::endl;
            mpi_store_vec_to_file(filepath_out, pvals, S, M);
        }
        else if (rank == 0) {
            std::cout << "FATAL: unknown --pval-method \"" << pval_method << "\" (use se or loo)" << std::endl;
        }
    }
    else if (rank == 0) {
        std::cout << "FATAL: unknown --run-mode \"" << opt.get_run_mode() << "\" (use infere, test or association_test)" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
