#pragma once
#include <string>
#include <vector>

class Options {

public:
    Options() = default;
    Options(int argc, char** argv) {
        read_command_line_options(argc, argv);
        check_options();
    }

    void read_command_line_options(int argc, char** argv);

    std::string get_meth_file() const { return meth_file; }
    std::string get_meth_file_test() const { return meth_file_test; }
    std::string get_phen_file() const { return phen_file; }
    std::string get_phen_file_test() const { return phen_file_test; }
    std::string get_true_signal_file() const { return true_signal_file; }
    std::string get_estimate_file() const { return estimate_file; }
    std::string get_r1_file() const { return r1_file; }
    std::string get_cov_estimate_file() const { return cov_estimate_file; }
    std::string get_cov_file() const { return cov_file; }
    std::string get_cov_file_test() const { return cov_file_test; }

    std::string get_out_dir() const { return out_dir; }
    std::string get_out_name() const { return out_name; }
    std::string get_model() const { return model; }
    std::string get_run_mode() const { return run_mode; }
    std::string get_pval_method() const { return pval_method; }

    double get_stop_criteria_thr() const { return stop_criteria_thr; }
    double get_merge_vars_thr() const { return merge_vars_thr; }
    double get_EM_err_thr() const { return EM_err_thr; }
    double get_rho() const { return rho; }
    double get_probit_var() const { return probit_var; }

    unsigned int get_EM_max_iter() const { return EM_max_iter; }
    unsigned int get_CG_max_iter() const { return CG_max_iter; }
    double get_CG_err_tol() const {return CG_err_tol; }

    unsigned int get_Mt()  const { return Mt; }
    unsigned int get_Mt_test()  const { return Mt_test; }
    unsigned int get_N() const { return N; }
    unsigned int get_N_test() const { return N_test; }
    unsigned int get_num_mix_comp() const { return num_mix_comp; }
    unsigned int get_C() const { return C; }
    unsigned int get_redglob() const { return redglob; }
    unsigned int get_learn_vars() const { return learn_vars; }
    unsigned int get_learn_prior_delay() const { return learn_prior_delay; }
    double get_h2() const { return h2; }
    double get_gam1() const { return gam1; }
    double get_alpha_scale() const { return alpha_scale; }
    int  get_verbosity() const { return verbosity; }
    unsigned int get_iterations() const { return iterations; }

    std::vector<double> get_vars() const { return vars; }
    std::vector<double> get_probs() const { return probs; }
    std::vector<int> get_test_iter_range() const { return test_iter_range; }

private:
    std::string meth_file = "";
    std::string meth_file_test = "";
    std::string phen_file;
    std::string phen_file_test;
    std::string true_signal_file;
    std::string estimate_file = "";
    std::string r1_file = "";
    std::string cov_estimate_file = "";
    std::string cov_file = "";
    std::string cov_file_test = "";
    std::string run_mode = "infere";
    std::string out_dir = "";
    std::string out_name = "";
    std::string model = "linear";
    std::string pval_method = "se";

    double stop_criteria_thr = 0.01;
    double merge_vars_thr = 5e-1;
    double EM_err_thr = 1e-2;
    unsigned int EM_max_iter = 1;
    unsigned int CG_max_iter = 500;
    double CG_err_tol = 1e-5;
    unsigned int Mt;
    unsigned int N;
    unsigned int N_test;
    unsigned int Mt_test;
    unsigned int num_mix_comp = 10;
    unsigned int learn_vars = 1;
    unsigned int learn_prior_delay = 1;
    double alpha_scale = 1.0;
    unsigned int redglob = 0;
    unsigned int C = 0;
    double probit_var = 1;
    double rho = 0.5;
    double h2 = 0.5;
    double gam1 = 1e-6;
    int verbosity = 0;
    unsigned int iterations = 50;

    std::vector<double> vars = std::vector<double> {0, 1e-06, 6e-06, 3e-05, 2e-04, 1e-03, 6e-03, 3e-02, 2e-01, 1e+00};
    std::vector<double> probs = std::vector<double> {9.90000e-01, 5.00000e-03, 2.50000e-03, 1.25000e-03, 6.25000e-04, 3.12500e-04, 1.56250e-04, 7.81250e-05, 3.90625e-05, 3.90625e-05};
    std::vector<int> test_iter_range = std::vector<int>{1, 50};

    void fail_if_last(char** argv, const int i);
    void check_options();
};
