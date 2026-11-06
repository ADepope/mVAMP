#include "data.hpp"
#include <cmath>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <limits>
#include <regex>
#include <cassert>
#include <mpi.h>
#include <omp.h>
#include <immintrin.h>
#include "utilities.hpp"

//******************
//  CONSTRUCTORS
//******************

// -> DESCRIPTION:
//
//      constructor that is given a file pointer to the phenotype and marker file
//
data::data(std::string phenfp, std::string methfp, std::string data_class, const int N, const int M, const int Mt, const int S, const int rank, double alpha_scale) :
    Mt(Mt),
    N(N),
    M(M),
    S(S),
    rank(rank),
    phenfp(phenfp),
    methfp(methfp),
    data_class(data_class),
    alpha_scale(alpha_scale) {

    mave = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(mave, __LINE__, __FILE__);
    msig = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(msig, __LINE__, __FILE__);

    if (data_class == "bin_class")
        read_phen(false);
    else
        read_phen(true);

    read_methylation_data();
    compute_markers_statistics();
}

//**************************
// DATA LOADING PROCEDURES
//**************************

// -> DESCRIPTION:
//
//      Read phenotype file assuming PLINK format:
//      Family ID, Individual ID, Phenotype; One row per individual
//
void data::read_phen(bool standardize) {

    std::ifstream infile(phenfp);
    std::string line;
    std::regex re("\\s+");

    double sum = 0.0;

    if (infile.is_open()) {
        nonas = 0, nas = 0;
        while (getline(infile, line)) {

            std::sregex_token_iterator first{line.begin(), line.end(), re, -1}, last;
            std::vector<std::string> tokens{first, last};
            if (tokens[2] == "NA") {
                throw "NAN in data!";
            } else {
                nonas += 1;
                phen_data.push_back(atof(tokens[2].c_str()));
                sum += atof(tokens[2].c_str());
            }
        }
        infile.close();

        // fail if the input size does not match the size of the read data
        assert(nas + nonas == N);

        // Center and scale
        if (standardize){
            double avg = sum / double(nonas);

            double sqn = 0.0;
            for (int i=0; i<phen_data.size(); i++) {
                if (phen_data[i] != std::numeric_limits<double>::max()) {
                    sqn += (phen_data[i] - avg) * (phen_data[i] - avg);
                }
            }
            sqn = sqrt( double(nonas-1) / sqn );
            for (int i=0; i<phen_data.size(); i++)
                phen_data[i] *= sqn;

            // saving intercept and scale term for phenotypes
            intercept = avg;
            scale = sqn; // inverse standard deviation
        }

    } else {
        std::cout << "FATAL: could not open phenotype file: " << phenfp << std::endl;
        exit(EXIT_FAILURE);
    }
}

// -> DESCRIPTION:
//
//      Reads methylation design matrix assuming matrix of doubles stores in binary format
//
void data::read_methylation_data(){

    double ts = MPI_Wtime();

    MPI_File methfh;

    if (rank == 0)
        std::cout << "meth file name = " << methfp.c_str() << std::endl;

    check_mpi(MPI_File_open(MPI_COMM_WORLD, methfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &methfh),  __LINE__, __FILE__);

    const size_t size_bytes = size_t(M) * size_t(N) * sizeof(double);

    meth_data = (double*)_mm_malloc(size_bytes, 64);

    printf("INFO  : rank %d has allocated %zu bytes (%.3f GB) for raw data.\n", rank, size_bytes, double(size_bytes) / 1.0E9);

    // Offset to section of the data file to be processed by task
    MPI_Offset offset = size_t(0) + size_t(S) * size_t(N) * sizeof(double);

    // Gather the sizes to determine common number of reads
    size_t max_size_bytes = 0;
    check_mpi(MPI_Allreduce(&size_bytes, &max_size_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

    const int NREADS = size_t( ceil(double(max_size_bytes)/double(INT_MAX/2)) );
    size_t bytes = 0;
    mpi_file_read_at_all <double*> (size_t(M) * size_t(N), offset, methfh, MPI_DOUBLE, NREADS, meth_data, bytes);

    MPI_File_close(&methfh);

    double te = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        std::cout <<"reading methylation data took " << te - ts << " seconds."<< std::endl;

}

// -> DESCRIPTION:
//
//      Reads covariates files, used with a probit model
//
void data::read_covariates(std::string covfp, int C){ // values should be separate with space delimiter

    if (C==0)
        return;

    double start = MPI_Wtime();

    std::ifstream covf(covfp);
    std::string line;
    std::regex re("\\s+");

    int i = 0;

    while (std::getline(covf, line)) // read the current line
    {
        i++;
        if(i == 1) continue; //skip header

        int Cobs = 0;
        std::vector<double> entries;
        std::sregex_token_iterator iter(line.begin(), line.end(), re, -1);
        std::sregex_token_iterator re_end;

        ++iter; // skip individual ID
        ++iter; // skip family ID

        for ( ; iter != re_end; ++iter){
            entries.push_back(std::stod(*iter));
            Cobs++;
        }

        if (Cobs != C){
            std::cout << "FATAL: number of covariates = " << Cobs << " does not match to the specified number of covariates = " << C << std::endl;
            exit(EXIT_FAILURE);
        }

        covs.push_back(entries);
    }

    double end = MPI_Wtime();

    if (rank == 0)
        std::cout << "rank = " << rank << ": reading covariates took " << end - start << " seconds to run." << std::endl;

    // Normalize covariates
    for(int covi = 0; covi < C; covi++){

        long double cavg = 0.0;
        long double csig = 0.0;

        for (int i = 0; i < N; i++) {
            cavg += covs[i][covi];
        }
        cavg = cavg / double(N);

        for (int i = 0; i < N; i++) {
            csig += ((covs[i][covi] - cavg) * (covs[i][covi] - cavg));
        }
        csig = sqrt(csig / double(N));

        for (int i = 0; i < N; i++) {
            if(csig < 0.00000001)
                covs[i][covi] = 0;
            else
                covs[i][covi] = (covs[i][covi] - cavg) / csig;
        }
    }
}

// -> DESCRIPTION:
//
//      Compute mean and associated standard deviation for markers
//
void data::compute_markers_statistics() {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double* mave = get_mave();
    double* msig = get_msig();

    double start = MPI_Wtime();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<M; i++) {
        size_t methix = size_t(i) * size_t(N);
        const double* methm = &meth_data[methix];
        double suma = 0.0;

#ifdef _OPENMP
#pragma omp simd reduction(+:suma)
#endif
        for (int j = 0; j < N; j++)
            suma += methm[j];

        //calculating vector of marker precision
        mave[i] = suma / double( get_nonas() );

        double sumsqr = 0.0;

#ifdef _OPENMP
#pragma omp simd reduction(+:sumsqr)
#endif
        for (int j=0; j<N; j++) {
            double val = (methm[j] - mave[i]);
            sumsqr += val * val;
        }

        if (sumsqr != 0.0)
            if (alpha_scale == 1.0)
                msig[i] = 1.0 / sqrt(sumsqr / (double( get_nonas() ) - 1.0));
            else
                msig[i] = 1.0 / pow( sqrt(sumsqr / (double( get_nonas() ) - 1.0)), alpha_scale );
        else
            msig[i] = 1.0;
    }

    double end = MPI_Wtime();
    if (rank == 0)
        std::cout << "rank = " << rank << ": statistics took " << end - start << " seconds to run." << std::endl;
}

//**************************************************************
// PROCEDURES IMPLEMENTING OPERATIONS INVOLVING DESIGN MATRIX A
//**************************************************************


// -> DESCRIPTION:
//
//      Computes < phen, (A_mloc - mu) * sigma_inv >
//
double data::dot_product(const int mloc, double* __restrict__ phen, const double mu, const double sigma_inv) {
  // always takes all individuals as their number is typically small in methylation studies

    double* meth = &meth_data[mloc * N];
    double dpa = 0.0;

#ifdef _OPENMP
#pragma omp parallel for simd schedule(static) reduction(+:dpa)
#endif
    for (int i=0; i<N; i++)
        dpa += (meth[i] - mu) * phen[i];

    return sigma_inv * dpa;
}


// -> DESCRIPTION:
//
//      Computes A^T phen using dot_product function above.
//

std::vector<double> data::ATx(double* __restrict__ phen) {

    std::vector<double> ATx(M, 0.0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int mloc=0; mloc < M; mloc++)
         ATx[mloc] = dot_product(mloc, phen, mave[mloc], msig[mloc]);

    // for stability reasons we scale design matrix A with 1/sqrt(number of people)
    double scale = 1.0 / sqrt(N);

    for (int mloc=0; mloc<M; mloc++)
        ATx[mloc] *= scale;

    return ATx;
}

// -> DESCRIPTION:
//
//      Computes A * phen using dot_product function above.
//

std::vector<double> data::Ax(double* __restrict__ phen) {

 // always takes all individuals as their number is typically small in methylation studies

    double* mave = get_mave();
    double* msig = get_msig();
    double* meth;
    std::vector<double> Ax_temp(N, 0.0);

    for (int i=0; i<M; i++){

        meth = &meth_data[i * N];

        double ave = mave[i];
        double sig_phen_i = msig[i] * phen[i];

#ifdef _OPENMP
#pragma omp parallel for simd shared(Ax_temp)
#endif
        for (int j=0; j<N; j++)
            Ax_temp[j] += (meth[j] - ave) * sig_phen_i;

    }

    // collecting results from different nodes
    std::vector<double> Ax_total(N, 0.0);

    MPI_Allreduce(Ax_temp.data(), Ax_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for(int i = 0; i < N; i++)
        Ax_total[i] /= sqrt(N);

    return Ax_total;
}

std::vector<double> data::Zx(std::vector<double> phen){

    std::vector<double> Zx_temp(N, 0.0);

    for (int i=0; i<N; i++)
        Zx_temp[i] = inner_prod(covs[i], phen, 0);

    return Zx_temp;
}

std::vector<double> data::pvals_loo(std::vector<double> z1, std::vector<double> y, std::vector<double> x1_hat){

    std::vector<double> pvals(M,0.0);
    std::vector<double> y_mod(N, 0.0); // phenotypic values corrected for genetic predictors

    for (int i=0; i<N; i++)
        y_mod[i] = y[i] - z1[i];

    for (int j=0; j<M; j++){

        std::vector<double> y_mark = y_mod;

        size_t methix = size_t(j) * size_t(N);
        const double* meth = &meth_data[methix];
        double sumx = 0.0, sumsqx = 0.0, sumxy = 0.0, sumy = 0.0, sumsqy = 0.0;

        for (int i=0; i<N; i++)
            y_mark[i] += meth[i] / sqrt(N) * x1_hat[j];

        for (int i=0; i<N; i++) {
            sumx += meth[i];
            sumsqx += meth[i]*meth[i];
            sumxy += meth[i]*y_mark[i];
            sumy += y_mark[i];
            sumsqy += y_mark[i] * y_mark[i];
        }
        pvals[j] = linear_reg1d_pvals(sumx, sumsqx, sumxy, sumy, sumsqy, N);
    }
    return pvals;
}
