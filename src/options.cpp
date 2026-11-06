#include <iostream>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <mpi.h>
#include "options.hpp"

// Function to parse command line options
void Options::read_command_line_options(int argc, char** argv) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::stringstream ss;
    ss << "\nmVAMP command line options:\n";

    for (int i=1; i<argc; ++i) {

        if (!strcmp(argv[i], "--meth-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            meth_file = argv[++i];
            ss << "--meth-file " << meth_file << "\n";
        }
        else if (!strcmp(argv[i], "--cov-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            cov_file = argv[++i];
            ss << "--cov-file " << cov_file << "\n";
        }
        else if (!strcmp(argv[i], "--cov-file-test")) {
            if (i == argc - 1) fail_if_last(argv, i);
            cov_file_test = argv[++i];
            ss << "--cov-file-test " << cov_file_test << "\n";
        }
        else if (!strcmp(argv[i], "--meth-file-test")) {
            if (i == argc - 1) fail_if_last(argv, i);
            meth_file_test = argv[++i];
            ss << "--meth-file-test " << meth_file_test << "\n";
        }
        else if (!strcmp(argv[i], "--estimate-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            estimate_file = argv[++i];
            ss << "--estimate-file " << estimate_file << "\n";
        }
        else if (!strcmp(argv[i], "--r1-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            r1_file = argv[++i];
            ss << "--r1-file " << r1_file << "\n";
        }
        else if (!strcmp(argv[i], "--cov-estimate-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            cov_estimate_file = argv[++i];
            ss << "--cov-estimate-file " << cov_estimate_file << "\n";
        }
        else if (!strcmp(argv[i], "--run-mode")) {
            if (i == argc - 1) fail_if_last(argv, i);
            run_mode = argv[++i];
            ss << "--run-mode " << run_mode << "\n";
        }
        else if (!strcmp(argv[i], "--phen-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            phen_file = argv[++i];
            ss << "--phen-file " << phen_file << "\n";
        }
        else if (!strcmp(argv[i], "--true-signal-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            true_signal_file = argv[++i];
            ss << "--true-signal-file " << true_signal_file << "\n";
        }
        else if (!strcmp(argv[i], "--phen-file-test")) {
            if (i == argc - 1) fail_if_last(argv, i);
            phen_file_test = argv[++i];
            ss << "--phen-file-test " << phen_file_test << "\n";
        }
        else if (!strcmp(argv[i], "--vars")) {
            if (i == argc - 1) fail_if_last(argv, i);
            std::string cslist = argv[++i];
            ss << "--vars " << cslist << "\n";
            std::stringstream sslist(cslist);
            std::string value;
            vars.clear();
            while (getline(sslist, value, ',')) {
                vars.push_back(atof(value.c_str()));
            }
        }
        else if (!strcmp(argv[i], "--probs")) {
            if (i == argc - 1) fail_if_last(argv, i);
            std::string cslist = argv[++i];
            ss << "--probs " << cslist << "\n";
            std::stringstream sslist(cslist);
            std::string value;
            probs.clear();
            while (getline(sslist, value, ',')) {
                probs.push_back(atof(value.c_str()));
            }
        }
        else if (!strcmp(argv[i], "--test-iter-range")) {
            if (i == argc - 1) fail_if_last(argv, i);
            std::string cslist = argv[++i];
            ss << "--test-iter-range " << cslist << "\n";
            std::stringstream sslist(cslist);
            std::string value;
            int nit = 0;
            while (getline(sslist, value, ',')) {
                test_iter_range[nit] = atoi(value.c_str());
                nit++;
            }
        }
        else if (!strcmp(argv[i], "--verbosity")) {
            if (i == argc - 1) fail_if_last(argv, i);
            verbosity = atoi(argv[++i]);
            ss << "--verbosity " << verbosity << "\n";
        }

        else if (!strcmp(argv[i], "--learn-vars")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 0) {
                std::cout << "FATAL  : option --learn-vars has to be a non-negative integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            learn_vars = (unsigned int) atoi(argv[++i]);
            ss << "--learn-vars " << learn_vars << "\n";
        }
        else if (!strcmp(argv[i], "--learn-prior-delay")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 0) {
                std::cout << "FATAL  : option --learn-prior-delay has to be a non-negative integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            learn_prior_delay = (unsigned int) atoi(argv[++i]);
            ss << "--learn-prior-delay " << learn_prior_delay << "\n";
        }
        else if (!strcmp(argv[i], "--iterations")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --iterations has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            iterations = (unsigned int) atoi(argv[++i]);
            ss << "--iterations " << iterations << "\n";
        }
        else if (!strcmp(argv[i], "--num-mix-comp")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --num-mix-comp has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            num_mix_comp = (unsigned int) atoi(argv[++i]);
            ss << "--num-mix-comp " << num_mix_comp << "\n";
        }
        else if (!strcmp(argv[i], "--out-dir")) {
            if (i == argc - 1) fail_if_last(argv, i);
            out_dir = argv[++i];
            ss << "--out-dir " << out_dir << "\n";
        }
        else if (!strcmp(argv[i], "--out-name")) {
            if (i == argc - 1) fail_if_last(argv, i);
            out_name = argv[++i];
            ss << "--out-name " << out_name << "\n";
        }
        else if (!strcmp(argv[i], "--model")) {
            if (i == argc - 1) fail_if_last(argv, i);
            model = argv[++i];
            ss << "--model " << model << "\n";
        }
        else if (!strcmp(argv[i], "--stop-criteria-thr")) {
            if (i == argc - 1) fail_if_last(argv, i);
            stop_criteria_thr = atof(argv[++i]);
            ss << "--stop-criteria-thr " << stop_criteria_thr << "\n";
        }
        else if (!strcmp(argv[i], "--merge-vars-thr")) {
            if (i == argc - 1) fail_if_last(argv, i);
            merge_vars_thr = atof(argv[++i]);
            ss << "--merge-vars-thr " << merge_vars_thr << "\n";
        }
        else if (!strcmp(argv[i], "--EM-err-thr")) {
            if (i == argc - 1) fail_if_last(argv, i);
            EM_err_thr = atof(argv[++i]);
            ss << "--EM-err-thr " << EM_err_thr << "\n";
        }
        else if (!strcmp(argv[i], "--alpha-scale")) {
            if (i == argc - 1) fail_if_last(argv, i);
            alpha_scale = atof(argv[++i]);
            ss << "--alpha-scale " << alpha_scale << "\n";
        }
        else if (!strcmp(argv[i], "--rho")) {
            if (i == argc - 1) fail_if_last(argv, i);
            rho = atof(argv[++i]);
            ss << "--rho " << rho << "\n";
        }
        else if (!strcmp(argv[i], "--probit-var")) {
            if (i == argc - 1) fail_if_last(argv, i);
            probit_var = atof(argv[++i]);
            ss << "--probit-var " << probit_var << "\n";
        }
        else if (!strcmp(argv[i], "--h2")) {
            if (i == argc - 1) fail_if_last(argv, i);
            h2 = atof(argv[++i]);
            ss << "--h2 " << h2 << "\n";
        }
        else if (!strcmp(argv[i], "--gam1")) {
            if (i == argc - 1) fail_if_last(argv, i);
            gam1 = atof(argv[++i]);
            ss << "--gam1 " << gam1 << "\n";
        }
        else if (!strcmp(argv[i], "--EM-max-iter")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --EM-max-iter has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            EM_max_iter = (unsigned int) atoi(argv[++i]);
            ss << "--EM-max-iter " << EM_max_iter << "\n";
        }
        else if (!strcmp(argv[i], "--Mt")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --Mt has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            Mt = (unsigned int) atoi(argv[++i]);
            ss << "--Mt " << Mt << "\n";
        }
        else if (!strcmp(argv[i], "--C")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 0) {
                std::cout << "FATAL  : option --C has to be a non-negative integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            C = (unsigned int) atoi(argv[++i]);
            ss << "--C " << C << "\n";
        }
        else if (!strcmp(argv[i], "--N")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --N has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            N = (unsigned int) atoi(argv[++i]);
            ss << "--N " << N << "\n";
        }
        else if (!strcmp(argv[i], "--N-test")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --N-test has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            N_test = (unsigned int) atoi(argv[++i]);
            ss << "--N-test " << N_test << "\n";
        }
        else if (!strcmp(argv[i], "--Mt-test")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --Mt-test has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            Mt_test = (unsigned int) atoi(argv[++i]);
            ss << "--Mt-test " << Mt_test << "\n";
        }
        else if (!strcmp(argv[i], "--CG-max-iter")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --CG-max-iter has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            CG_max_iter = (unsigned int) atoi(argv[++i]);
            ss << "--CG-max-iter " << CG_max_iter << "\n";
        }
        else if (!strcmp(argv[i], "--CG-err-tol")) {
            if (i == argc - 1) fail_if_last(argv, i);
            CG_err_tol = atof(argv[++i]);
            ss << "--CG-err-tol " << CG_err_tol << "\n";
        }
        else if (!strcmp(argv[i], "--pval-method")) {
            if (i == argc - 1) fail_if_last(argv, i);
            pval_method = argv[++i];
            ss << "--pval-method " << pval_method << "\n";
        }
        else {
            std::cout << "FATAL: option \"" << argv[i] << "\" unknown\n";
            exit(EXIT_FAILURE);
        }
    }

    if (rank == 0)
        std::cout << ss.str() << std::endl;
}

// Catch missing argument on last passed option
void Options::fail_if_last(char** argv, const int i) {
    std::cout << "FATAL  : missing argument for last option \"" << argv[i] <<"\". Please check your input and relaunch." << std::endl;
    exit(EXIT_FAILURE);
}

// Check for minimal setup: a methylation data file
void Options::check_options() {
    if (get_meth_file() == "" && get_meth_file_test() == "") {
        std::cout << "FATAL  : no meth file provided! Please use the --meth-file option." << std::endl;
        exit(EXIT_FAILURE);
    }
}