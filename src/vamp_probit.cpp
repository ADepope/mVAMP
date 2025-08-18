#include <vector>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <cmath>
#include <numeric> // contains std::accumulate
#include <random>
#include <omp.h>
#include <fstream>
#include <cfloat>
#include "vamp.hpp"
#include "utilities.hpp"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>


std::vector<double> vamp::infere_bin_class( data* dataset ){

    std::vector<double> metrics(12,0); // Error metrics from denoising and lmmse steps to save to csv [TP, TN, FP, FN, ACC, x1_corr | TP, TN, FP, FN, ACC, x2_corr]
    std::vector<double> params(8,0); // Parameters from denoising and lmmse steps to save to csv [alpha1 | beta1 | gam1 | tau1 | alpha2 | beta2 | gam2 | tau2 ]

    double total_time = 0;
    double tol = 1e-11;

    std::vector<double> x1_hat_d(M, 0.0);
    std::vector<double> x1_hat_d_prev(M, 0.0);
    std::vector<double> r1_prev(M, 0.0);
    std::vector<double> p1_prev(N, 0.0);
    std::vector<double> x1_hat_prev(M, 0.0);

    // tau1 = gam1; // hardcoding initial variability
    //tau1 = 1e-1; // ideally sqrt( tau_10 / v ) approx 1, since it is to be composed with a gaussian CDF
    tau1 = gam1;
    // double tau2_prev = 0;
    double gam1_prev = gam1;
    double tau1_prev = tau1;
    double gam1_max = gam1;

    double sqrtN = sqrt(N);

    std::vector<double> true_signal_scaled = true_signal;
    for (int i = 0; i < true_signal_scaled.size(); i++)
        true_signal_scaled[i] = true_signal[i] * sqrtN;
    std::vector<double> true_g = (*dataset).Ax(true_signal_scaled.data());

    // Bernoulli distribution for trace estimation
    std::bernoulli_distribution bern(0.5);
    bern_vec = std::vector<double> (M, 0.0);

    // Gaussian noise start
    p1 = simulate(N, std::vector<double> {1.0}, std::vector<double> {1.0});

    r1 = std::vector<double> (M, 0.0);
    r2 = r1;
    alpha1 = 0;

    // initializing z1_hat and p2
    z1_hat = std::vector<double> (N, 0.0);
    p2 = std::vector<double> (N, 0.0);
    if (C>0)
        cov_eff = std::vector<double> (C, 0.0);

    std::vector< std::vector<double> > Z = (*dataset).get_covs();
    std::vector<double> gg;

    for (int it = 1; it <= max_iter; it++)
    {    
        double start_covar = 0, stop_covar = 0, start_denoising = 0, stop_denoising = 0; // meassuring time

        if (rank == 0)
            std::cout << std::endl << "********************" << std::endl << "iteration = "<< it << std::endl << "********************" << std::endl;

        // Compute covariate effects
        if (rank == 0)
            std::cout << "...calculating covariate effects" << std::endl;
        if (it == 1 && C > 0){
            start_covar = MPI_Wtime();
            gg = z1_hat;
            //cov_eff = grad_desc_cov(y, gg, probit_var, Z, cov_eff); // std::vector<double>(C, 0.0)
            cov_eff = Newton_method_cov(y, gg, Z, cov_eff);

            if (rank == 0){
                for (int i0=0; i0<C; i0++){
                    std::cout << "cov_eff[" << i0 << "] = " << cov_eff[i0] << ", ";
                    if (i0 % 4 == 3)
                        std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            stop_covar = MPI_Wtime();
            if (rank == 0)
                std::cout << "Computing covariate effects took " << stop_covar - start_covar << " seconds." << std::endl;
        }
        
        // ---------- DENOISING x ---------- //

        start_denoising = MPI_Wtime();

        if (rank == 0)
            std::cout << "->DENOISING" << std::endl;

        x1_hat_prev = x1_hat;
        double alpha1_prev = alpha1;
        //double gam1_reEst_prev;
        //int it_revar = 1;

        //for (; it_revar <= auto_var_max_iter; it_revar++){

            // new signal estimate
            for (int i = 0; i < M; i++)
                x1_hat[i] = g1(r1[i], gam1);

            std::vector<double> x1_hat_m_r1 = x1_hat;
            for (int i0 = 0; i0 < x1_hat_m_r1.size(); i0++)
                x1_hat_m_r1[i0] = x1_hat_m_r1[i0] - r1[i0];

            // new MMSE estimate
            double sum_d = 0;
            for (int i=0; i<M; i++)
            {
                x1_hat_d[i] = g1d(r1[i], gam1);
                sum_d += x1_hat_d[i];
            }

            alpha1 = 0;
            MPI_Allreduce(&sum_d, &alpha1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            alpha1 /= Mt;
            eta1 = gam1 / alpha1;

            //if (it <= 1)
            //    break;

            //gam1_reEst_prev = gam1;
            //gam1 = std::min( std::max(  1 / (1/eta1 + l2_norm2(x1_hat_m_r1, 1)/Mt), gamma_min ), gamma_max );

            if (it > learn_prior_delay)
                updatePrior();

            // prior distribution parameters
            if (rank == 0){
                std::cout << "Prior variances = ";
                for (int i = 0; i < vars.size(); i++)
                    std::cout << vars[i] / (double) N << ' ';
                std::cout << std::endl;
    
                std::cout << "Prior probabilities = ";
                for (int i = 0; i < probs.size(); i++)
                    std::cout << probs[i] << ' ';
                std::cout << std::endl;
            }

            //if ( abs(gam1 - gam1_reEst_prev) < 1e-3 )
            //    break;
            
        //}

        // damping on the level of x1
        if (it > 1){ 
            for (int i = 0; i < M; i++)
                x1_hat[i] = rho * x1_hat[i] + (1-rho) * x1_hat_prev[i];
            
            alpha1 = rho * alpha1 + (1-rho) * alpha1_prev;
        }

        // saving x1_hat to file
        std::vector<double> x1_hat_scaled = x1_hat;
        std::string filepath_out = out_dir + "/" + out_name + "_it_" + std::to_string(it) + ".bin";
        int S = (*dataset).get_S();
        for (int i0 = 0; i0 < x1_hat_scaled.size(); i0++)
            x1_hat_scaled[i0] =  x1_hat[i0] / sqrtN;
        mpi_store_vec_to_file(filepath_out, x1_hat_scaled, S, M);

        if (rank == 0)
            std::cout << "...storing x1_hat to file " << filepath_out << std::endl;

        // saving r1 to file
        std::string filepath_out_r1 = out_dir + "/" + out_name + "_r1_it_" + std::to_string(it) + ".bin";
        std::vector<double> r1_scaled = r1;
        for (int i0 = 0; i0 < r1_scaled.size(); i0++)
            r1_scaled[i0] =  r1[i0] / sqrtN;
        mpi_store_vec_to_file(filepath_out_r1, r1_scaled, S, M);

        if (rank == 0)
            std::cout << "...storing r1 to file " << filepath_out_r1 << std::endl;

        // Error meassure
        double x1_corr = inner_prod(x1_hat, true_signal_scaled, 1) / sqrt( l2_norm2(x1_hat, 1) * l2_norm2(true_signal_scaled, 1) );

        MPI_Barrier(MPI_COMM_WORLD);
    
        gam_before = gam2;
        gam2 = std::min( std::max( eta1 - gam1, gamma_min ), gamma_max );

        std::vector<double> r2_prev = r2;
        for (int i = 0; i < M; i++)
            r2[i] = (eta1 * x1_hat[i] - gam1 * r1[i]) / gam2;

        // ---------- DENOISING Z ----------- //
    
        std::vector<double> y = (*dataset).get_phen();

        std::vector<double> z1_hat_prev = z1_hat;
        //double tau1_reEst_prev;
        //it_revar = 1;
        //auto_var_max_iter = 1;
        double beta1;

        //for (; it_revar <= auto_var_max_iter; it_revar++){

            // new signal estimate
            for (int i=0; i<N; i++){
                double m_cov = 0;
                if (C>0)
                    m_cov = inner_prod(Z[i], cov_eff, 0);

                z1_hat[i] = g1_bin_class(p1[i], tau1, y[i], m_cov);
            }

            std::vector<double> z1_hat_m_p1 = z1_hat;
            for (int i0 = 0; i0 < N; i0++)
                z1_hat_m_p1[i0] = z1_hat_m_p1[i0] - p1[i0];

            // new MMSE estimate
            beta1 = 0;
            for (int i=0; i<N; i++){
                double m_cov = 0;
                if (C>0)
                    m_cov = inner_prod(Z[i], cov_eff, 0);

                beta1 += g1d_bin_class(p1[i], tau1, y[i], m_cov);
            }
            if(beta1 >= N)
                beta1 = N - 1.0;
            beta1 /= N;
            double zeta1 = tau1 / beta1;

            //if (it <= 1)
            //    break;

            //tau1_reEst_prev = tau1;
            //tau1 = std::min( std::max(  1 / (1/zeta1 + l2_norm2(z1_hat_m_p1, 0)/N), gamma_min ), gamma_max );

            //if ( abs(tau1 - tau1_reEst_prev) < 1e-2 )
            //    break;
            
        //}
        
        for (int i=0; i<N; i++)
            p2[i] = (z1_hat[i] - beta1*p1[i]) / (1-beta1);

        tau2 = tau1 * (1-beta1) / beta1;

        // Save and print hyperparameters
        params[0] = alpha1;
        params[1] = beta1;
        params[2] = gam1;
        params[3] = tau1;

        if(rank == 0){
            std::cout << "alpha1 = " << alpha1 << std::endl;
            std::cout << "beta1 = " << beta1 << std::endl;
            //std::cout << "gam1 = " << gam1 << std::endl;
            //std::cout << "eta1 = " << eta1 << std::endl;
            std::cout << "tau1 = " << tau1 << std::endl;
        }

        // ------ Error metrics ------- //
        // Predictions
        std::vector<double> z1hat = (*dataset).Ax(x1_hat_scaled.data());
        std::vector<double> y1_hat = predict_probit(z1hat, 0.5);
        std::vector<int> conf_mat_1 = confusion_matrix(y, y1_hat);

        double acc1 = accuracy(conf_mat_1); // (TP + TN) / (TP + TN + FP + FN)

        metrics[0] = conf_mat_1[0];
        metrics[1] = conf_mat_1[1];
        metrics[2] = conf_mat_1[2];
        metrics[3] = conf_mat_1[3];
        metrics[4] = acc1;
        metrics[5] = x1_corr;

        if (rank == 0){
            std::cout <<  "Corr(x1_hat,x0) = " << x1_corr <<  std::endl;
            std::cout <<  "Accuracy1 = " << acc1 <<  std::endl;
        }

        stop_denoising = MPI_Wtime();

        // ---------- LMMSE estimation of x ---------- //
        double start_LMMSE = MPI_Wtime();
        if (rank == 0)
            std::cout << std::endl << "->LMMSE" << std::endl;

        // Sample random Bernoulli vector for trace estimation
        for (int i = 0; i < M; i++)
            bern_vec[i] = (2*bern(rd) - 1) / sqrt(Mt); // Bernoulli variables are sampled independently

        std::vector<double> v = (*dataset).ATx(p2.data());

        for (int i = 0; i < M; i++)
            v[i] = tau2 * v[i] + gam2 * r2[i];

        // running conjugate gradient solver to compute LMMSE
        std::vector<double> x2_hat_prev = x2_hat;
        x2_hat = precondCG_solver(v, std::vector<double> (M, 0.0), tau2, 1, dataset); // precond_change!
        
        // Onsager
        double alpha2_prev = alpha2;
        double alpha2 = g2d_onsager(gam2, tau2, dataset);

        //for (int i = 0; i < M; i++)
        //    x2_hat[i] = rho * x2_hat[i] + (1 - rho) * x2_hat_prev[i];
        //alpha2 = rho * alpha2 + (1 - rho) * alpha2_prev;

        // Scaling x2_hat estimates
        std::vector<double> x2_hat_s = x2_hat;
        for (int i=0; i<x2_hat_s.size(); i++)
            x2_hat_s[i] = x2_hat[i] / sqrt(N);

        // Error measures
        // correlation Corr(x2_hat, x0)
        double x2_corr = inner_prod(x2_hat, true_signal_scaled, 1) / sqrt( l2_norm2(x2_hat, 1) * l2_norm2(true_signal_scaled, 1) );

        eta2 = gam2 / alpha2;

        // re-estimation of gam2
        //std::vector<double> x2_hat_m_r2 = x2_hat;
        //for (int i0 = 0; i0 < x2_hat_m_r2.size(); i0++)
        //    x2_hat_m_r2[i0] = x2_hat_m_r2[i0] - r2[i0];

        //if (it > 1)
        //    gam2 = std::min( std::max(  1 / (1/eta2 + l2_norm2(x2_hat_m_r2, 1)/Mt), gamma_min ), gamma_max );

        r1_prev = r1;
        for (int i=0; i<M; i++)
            r1[i] = (x2_hat[i] - alpha2 * r2[i]) / (1 - alpha2);

        // Damping
        //for (int i=0; i<M; i++)
        //    r1[i] = rho * r1[i] + (1-rho) * r1_prev[i];

        gam1_prev = gam1;
        gam1 = gam2 * (1-alpha2) / alpha2;
        gam1 = std::min(std::max(gam1, gamma_min ), gamma_max);

        // apply damping 
        //gam1 = rho * gam1 + (1-rho) * gam1_prev;

        // ---------- LMMSE estimation of z ----------- //
        z2_hat = (*dataset).Ax(x2_hat.data());
        
        double beta2 = (double) Mt / N * (1 - alpha2);

        // re-estimation of tau2
        //std::vector<double> z2_hat_m_p2 = z2_hat;
        //for (int i0 = 0; i0 < N; i0++)
        //    z2_hat_m_p2[i0] = z2_hat_m_p2[i0] - p2[i0];

        //double zeta2 = tau2 / beta2;

        //if (it > 1)
        //    tau2 = 1.0 / (1.0/zeta2 + l2_norm2(z2_hat_m_p2, 0)/N);

        p1_prev = p1;
        for (int i=0; i<N; i++)
            p1[i] = (z2_hat[i] - beta2 * p2[i]) / (1-beta2);

        // Damping
        //for (int i=0; i<N; i++)
        //    p1[i] = rho_it * p1[i] + (1 - rho) * p1_prev[i];

        tau1_prev = tau1;
        tau1 = tau2 * (1 - beta2) / beta2;
        tau1 = std::min(std::max(tau1, gamma_min ), gamma_max);

        //  apply damping to z variance
        //tau1 = rho * tau1 + (1 - rho) * tau1_prev;

        // Save and print hyperparameters
        params[4] = alpha2;
        params[5] = beta2;
        params[6] = gam2;
        params[7] = tau2;

        if(rank == 0){
            std::cout << "alpha2 = " << alpha2 << std::endl;
            std::cout << "beta2 = " << beta2 << std::endl;
            std::cout << "gam1 = " << gam1 << std::endl;
            std::cout << "gam2 = " << gam2 << std::endl;
            //std::cout << "eta2 = " << eta2 << std::endl;
            std::cout << "tau2 = " << tau2 << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double stop_LMMSE = MPI_Wtime();

        total_time += (stop_LMMSE - start_LMMSE) + (stop_denoising - start_denoising) + (stop_covar - start_covar);

        // ------ERROR metrics ------- //  
        std::vector<double> z2hat = (*dataset).Ax(x2_hat_s.data());
        std::vector<double> y2_hat = predict_probit(z2hat, 0.5);

        std::vector<int> conf_mat_2 = confusion_matrix(y, y2_hat);

        double acc2 = accuracy(conf_mat_2);

        metrics[6] = conf_mat_2[0];
        metrics[7] = conf_mat_2[1];
        metrics[8] = conf_mat_2[2];
        metrics[9] = conf_mat_2[3];
        metrics[10] = acc2;
        metrics[11] = x2_corr;

        if (rank == 0){
            std::cout <<  "Corr(x2_hat, x0) = " << x2_corr <<  std::endl;
            std::cout <<  "Accuracy2 = " << acc2 <<  std::endl;
        }

        // Store mixture probabilities and variances
        std::vector<double> prior_params;
        prior_params.push_back(probs.size());
        for(int i=0; i < probs.size(); i++)
            prior_params.push_back(probs[i]);
        for(int i=0; i < vars.size(); i++)
            prior_params.push_back(vars[i]);

        if (rank == 0){
            std::cout << "...storing parameters to CSV files" << std::endl;
            write_ofile_csv(out_params_fh, it, &params);
            write_ofile_csv(out_metrics_fh, it, &metrics);
            write_ofile_csv(out_prior_fh, it, &prior_params);
        }

        if(rank == 0)
            std::cout << "Total time so far = " << total_time << " seconds." << std::endl;

        // Stopping criteria
        if (rank == 0)
            std::cout << "...stopping criteria assessment" << std::endl;

        std::vector<double> x1_hat_diff = std::vector<double>(M, 0.0);
        for (int i = 0; i < M; i++)
            x1_hat_diff[i] = x1_hat_prev[i] - x1_hat[i];

        double NMSE = sqrt( l2_norm2(x1_hat_diff, 1) / l2_norm2(x1_hat_prev, 1) );
        if (rank == 0)
            std::cout << "x1_hat NMSE = " << NMSE << std::endl;
        if (rank == 0)
            std::cout << "stop_criteria_thr = " << stop_criteria_thr << std::endl;

        if (it > 1 && NMSE < stop_criteria_thr){
            if (rank == 0)
                std::cout << "...stopping criteria fulfilled" << std::endl;
            break;
        }
        if (it == max_iter){
            if (rank == 0)
                std::cout << "...maximal number of iterations was achieved. The algorithm might not converge!" << std::endl;
        }
    }
    
    return x1_hat;

}

double vamp::g1_bin_class(double p, double tau1, double y, double m_cov = 0){

    double c = (p + m_cov) / sqrt(probit_var + 1.0/tau1);
    double temp;
    double normalPdf_normalCdf = 2.0 /  sqrt(2*M_PI) / erfcx( - (2*y-1) * c / sqrt(2) );   
    temp = p + (2*y-1) * normalPdf_normalCdf / tau1 / sqrt(probit_var + 1.0/tau1);

    return temp;

}

double vamp::g1d_bin_class(double p, double tau1, double y, double m_cov = 0){

    double c = (p + m_cov) / sqrt(probit_var + 1.0/tau1);
    double Nc_phiyc = 2.0 /  sqrt(2*M_PI) / erfcx( - (2*y-1) * c / sqrt(2) );
    double temp;
    temp = 1 -  Nc_phiyc / (1 + tau1 * probit_var) * ( (2*y-1)*c + Nc_phiyc ); // because der = tau * Var
    return temp;

}

double vamp::mlogL_probit(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta){

    double mlogL = 0;

#pragma omp parallel for reduction( + : mlogL )
    for (int i=0; i<N; i++){
        double g_i = gg[i] + inner_prod(Z[i], eta, 0);
        double arg = (2*y[i] - 1) / sqrt(probit_var) * g_i;
        double phi_arg = normal_cdf(arg);
        mlogL -= log(phi_arg);
    }
    return mlogL/N;
}

std::vector<double> vamp::grad_cov(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta){ 
    // gg = g_genetics, gc = g_covariates, eta = vector of covariate's effect sizes

    std::vector<double> grad(C, 0.0);

    for (int j=0; j<C; j++){

        for (int i=0; i<N; i++){
            double g_i = gg[i] + inner_prod(Z[i], eta, 0);
            double arg = (2*y[i] - 1) / sqrt(probit_var) * g_i;
            double ratio = 2.0 /  sqrt(2*M_PI) / erfcx( - arg / sqrt(2) );
            grad[j] += (-1) * ratio * (2*y[i]-1) / sqrt(probit_var) * Z[i][j]; // because we take a gradient of -logL, not of logL
        }
    }

    for (int j=0; j<C; j++)
        grad[j] /= N;

    return grad;
}

std::vector<double> vamp::Newton_method_cov(std::vector<double> y, std::vector<double> gg, std::vector< std::vector<double> > Z, std::vector<double> eta){

    using namespace boost::numeric::ublas;

    std::vector<double> eta_new;

    for (int it=0; it<=500; it++){

        matrix<double> WXm(N,C), Xtm(C,N);
        vector<double> lambda(N);

        for(int i=0; i<N; i++){

            double g_i = gg[i] + inner_prod(Z[i], eta, 0);
            double arg = (2*y[i] - 1) * g_i;
            double phi_arg = normal_cdf(arg);
            double ratio = 2.0 /  sqrt(2*M_PI) / erfcx( - arg / sqrt(2) );
            lambda(i) = ratio * (2*y[i]-1);
        
            for (int j=0; j<C; j++){
                Xtm(j,i) = Z[i][j];
                WXm(i,j) = Z[i][j] * lambda(i) * (lambda(i) + g_i);
            }
        }

        matrix<double> XtmZm = prod(Xtm, WXm);
        vector<double> RHS = prod(Xtm, lambda);    
        permutation_matrix<double> pm(XtmZm.size1());
        
        int sing = lu_factorize(XtmZm, pm);
            
        if (sing == 0)
            lu_substitute(XtmZm, pm, RHS);
        else
            RHS = vector<double>(C);

        eta_new = eta;
        std::vector<double> displ(C, 0.0);

        std::vector<double> grad = grad_cov(y, gg, probit_var, Z, eta);
        double scale = 1;
        double init_val = mlogL_probit(y, gg, probit_var, Z, eta);

        for (int i=1; i<300; i++){ // 0.9^300 = 1.8e-14

            for (int j=0; j<C; j++)
                displ[j] = scale * RHS(j);

            std::transform (eta.begin(), eta.end(), displ.begin(), eta_new.begin(), std::plus<double>());

            double curr_val = mlogL_probit(y, gg, probit_var, Z, eta_new);

            if (curr_val <= init_val + inner_prod(displ, grad,0)/2){
                if (rank == 0)
                    std::cout << "scale = " << scale << std::endl;
                break;
            }
            scale *= 0.9;
        }

        std::vector<double> diff = eta;
        for (int i=0; i<diff.size(); i++)
            diff[i] -= eta_new[i];
        double norm_eta = sqrt( l2_norm2(eta, 0) );
        double rel_err;
        if (norm_eta == 0)
            rel_err = 1;
        else
            rel_err = sqrt( l2_norm2(diff, 0) ) / norm_eta;

        if (rank == 0 && verbosity == 1)
            std::cout << "[Newton_cov] it = " << it <<", relative err = "<< rel_err << std::endl;
        if (rel_err < 1e-4){
            if (rank == 0)
                std::cout << "[Newton_cov] relative error <= 1e-4 - stoping criteria satisfied" << std::endl;
            break;
        }

        // another stopping criteria based on likelihood value
        init_val = mlogL_probit(y, gg, probit_var, Z, eta);
        eta = eta_new;
        double curr_val = mlogL_probit(y, gg, probit_var, Z, eta);

        if (curr_val > init_val){
            if (rank == 0){
                std::cout << "previous mlogL = " << init_val << ", current mlogL = " << curr_val << std::endl;
                std::cout << "likelihood value is not increasing -> terminating Newton-Raphson mehod" << std::endl;
            }
            break;
        }    
    }
    return eta;
}

std::vector<double> vamp::predict_probit(std::vector<double> z, double th){
    std::vector<double> y(N,0);
    for (int i = 0; i < N; i++){
        double prob = normal_cdf(z[i]);
        if (prob >= th)
            y[i] = 1;
        else 
            y[i] = 0;
    }
    return y;
}

std::vector<int> vamp::confusion_matrix(std::vector<double> y, std::vector<double> yhat){
    std::vector<int> conf_mat(4,0);
    int TP=0, TN=0, FP=0, FN=0;

    for (int i = 0; i < N; i++){
        if (y[i] == 1 && yhat[i] == 1)
            TP++;
        else if (y[i] == 0 && yhat[i] == 0)
            TN++;
        else if (y[i] == 1 && yhat[i] == 0)
            FN++;
        else if (y[i] == 0 && yhat[i] == 1)
            FP++;
    }

    conf_mat[0] = TP;
    conf_mat[1] = TN;
    conf_mat[2] = FP;
    conf_mat[3] = FN;

 return conf_mat;
}
double vamp::accuracy(std::vector<int> conf_mat){

    int TP = conf_mat[0];
    int TN = conf_mat[1];
    int FP = conf_mat[2];
    int FN = conf_mat[3];

    double acc = (double)(TP + TN) / (double)(TP + TN + FP + FN);

    return acc;
}

void vamp::probit_err_measures(data *dataset, int sync, std::vector<double> true_signal, std::vector<double> est, std::string var_name){
    
    // correlation
    double corr = inner_prod(est, true_signal, sync) / sqrt( l2_norm2(est, sync) * l2_norm2(true_signal, sync) );

    if ( rank == 0 )
        std::cout << "correlation " + var_name + " = " << corr << std::endl;  

    
    // l2 signal error
    int len = (int) ( N + (M-N) * sync );
    std::vector<double> temp(len, 0.0);

    for (int i = 0; i<len; i++)
        temp[i] = est[i] - true_signal[i];

    double l2_signal_err = sqrt( l2_norm2(temp, sync) / l2_norm2(true_signal, sync) );
    if (rank == 0)
        std::cout << "l2 signal error for " + var_name + " = " << l2_signal_err << std::endl;


    // precision calculation
    //double std = calc_stdev(temp, sync);

    //if (rank == 0)
    //    std::cout << "true precision for " + var_name + " = " << 1.0 / N / std / std << std::endl;


    // prior distribution parameters
    if (sync == 1){

        if (rank == 0)
        std::cout << "prior variances = ";

        for (int i = 0; i < vars.size(); i++)
            if (rank == 0)
                std::cout << vars[i] << ' ';

        if (rank == 0) {
            std::cout << std::endl;
            std::cout << "prior probabilities = ";
        }

        for (int i = 0; i < probs.size(); i++)
            if (rank == 0)
                std::cout << probs[i] << ' ';

        if (rank == 0)
            std::cout << std::endl;
    }
    
    
}