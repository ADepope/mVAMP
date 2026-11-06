#pragma once
#include <string>
#include <vector>
#include <memory>
#include <immintrin.h> // contains definition of _mm_malloc

// class data contains methylation and phenotypic data and auxiliary functions related
// to loading them, computing marker statistics and efficient products between
// the (standardized) methylation matrix and a vector.

class data {

private:

    int Mt;         // total number of markers
    int N;          // number of individuals
    int M;          // number of markers attributed to the node
    int S;          // marker starting index
    int rank;       // rank of MPI process

    std::string phenfp;       // filepath to phenotypes
    std::string methfp;          // filepath to methylation data
    std::string data_class;

    std::vector<double> phen_data;   // vector of phenotype data

    int nonas;
    int nas;

    double* mave     = nullptr;
    double* msig     = nullptr;

    std::vector<double> mave_people;
    std::vector<double> msig_people;
    std::vector<double> numb_people;

    double* meth_data = nullptr;    // methylation data, M x N, stored marker by marker
    std::vector< std::vector<double> > covs;

    double sigma_max = 1e8;
    double intercept = 0;
    double scale = 1;
    double alpha_scale;

public:

    // ACCESSING CLASS VARIABLES
    std::vector<double> get_phen(){ return phen_data; };
    double get_intercept() { return intercept; };
    double get_scale() { return scale; };
    double get_sigma_max() { return sigma_max; };

    double * get_meth_data() { return meth_data; }
    std::vector< std::vector<double> > get_covs() { return covs; }
    void set_phen(std::vector<double> new_data) { phen_data = new_data; }

    double* get_mave()        { return mave; }
    double* get_msig()        { return msig; }

    std::vector<double> get_mave_people() { return mave_people; };
    std::vector<double> get_msig_people() { return msig_people; };
    std::vector<double> get_numb_people() { return numb_people; };

    int     get_nonas()       { return nonas; }
    void    set_nonas(int num){ nonas = num; }
    int     get_S()     const { return S; }

    //  CONSTRUCTORS and DESTRUCTOR
    data(std::string phenfp, std::string methfp, std::string data_class, const int N, const int M, const int Mt, const int S, const int rank, double alpha_scale = 1);
    ~data() {
        if (mave      != nullptr) _mm_free(mave);
        if (msig      != nullptr) _mm_free(msig);
        if (meth_data != nullptr) _mm_free(meth_data);
    }
    data(const data&) = delete;             // owns raw buffers, so no copies
    data& operator=(const data&) = delete;

    // DATA LOADING PROCEDURES
    void read_phen(bool standardize);   // reading phenotype file
    void read_covariates(std::string covfp, int C = 0);
    void read_methylation_data();
    void compute_markers_statistics();
    void compute_people_statistics();

    // PROCEDURES IMPLEMENTING OPERATIONS INVOLVING DESIGN MATRIX A
    std::vector<double> Ax(double* __restrict__ phen);

    std::vector<double> ATx(double* __restrict__ phen);
    double dot_product(const int mloc, double* __restrict__ phen, const double mu, const double sigma_inv);
    std::vector<double> Zx(std::vector<double> phen);

    std::vector<double> pvals_loo(std::vector<double> z1, std::vector<double> y, std::vector<double> x1_hat);


};

