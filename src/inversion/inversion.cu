# include "inversion.cuh"

void Inversion::set_parameters()
{
    title = "\033[34mFull Waveform Inversion\033[0;0m";
    
    set_main_parameters();
    
    set_wavelet();
    set_geometry();    
    set_seismograms();
    set_abc_dampers();
    set_properties();
    set_coordinates();

    rbc_ratio = std::stof(catch_parameter("inv_rbc_ratio", parameters)); 
    rbc_varVp = std::stof(catch_parameter("inv_rbc_varVp", parameters)); 
    rbc_length = std::stof(catch_parameter("inv_rbc_length", parameters));

    abc_nb = nb;
    abc_nxx = nxx;
    abc_nzz = nzz;
    abc_matsize = matsize;

    rbc_nb = (int)(rbc_length / dh) + 1;

    rbc_nxx = nx + 2*rbc_nb;
    rbc_nzz = nz + 2*rbc_nb;
    rbc_matsize = rbc_nxx*rbc_nzz;

    vp = new float[nPoints]();
    
    reduce_boundary(Vp, vp);

    nb = rbc_nb;
    nxx = rbc_nxx;
    nzz = rbc_nzz;
    matsize = rbc_matsize;

    h_rbc_Vp = new float[rbc_matsize]();

    expand_boundary(vp, h_rbc_Vp);

    input_folder = catch_parameter("inversion_input_folder", parameters);
    output_folder = catch_parameter("inversion_output_folder", parameters);
    residuo_folder = catch_parameter("inversion_residuo_folder", parameters);

    max_iteration = std::stoi(catch_parameter("max_iteration", parameters));

    iteration = 0;

    sumPs = new float[nPoints]();
    gradient = new float[nPoints]();

    partial = new float[rbc_matsize]();

    cudaMalloc((void**)&(d_Pr), rbc_matsize*sizeof(float));
    cudaMalloc((void**)&(d_Ps), rbc_matsize*sizeof(float));
    
    cudaMalloc((void**)&(d_Psold), rbc_matsize*sizeof(float));
    cudaMalloc((void**)&(d_Prold), rbc_matsize*sizeof(float));
    
    cudaMalloc((void**)&(d_sumPs), rbc_matsize*sizeof(float));

    cudaMalloc((void**)&(d_gradient), rbc_matsize*sizeof(float));

    cudaMalloc((void**)&(d_rbc_Vp), rbc_matsize*sizeof(float));

    cudaMemcpy(d_rbc_Vp, h_rbc_Vp, rbc_matsize*sizeof(float), cudaMemcpyHostToDevice);
}

void Inversion::set_observed_data()
{
    obs_data = new float[nt*geometry->nTraces]();

    std::string input_file = input_folder + "seismogram_nt" + std::to_string(nt) + "_nTraces" + std::to_string(geometry->nTraces) + "_" + std::to_string((int)(fmax)) + "Hz_" + std::to_string(int(1e6f*dt)) + "us.bin";

    import_binary_float(input_file, obs_data, nt*geometry->nTraces); 
}

void Inversion::set_calculated_data()
{
    ABC = true;

    nb = abc_nb;
    nxx = abc_nxx;
    nzz = abc_nzz;
    matsize = abc_matsize;

    nBlocks = (int)((matsize + NTHREADS - 1) / NTHREADS);

    stage_info = "Calculating seismograms to compute residuo...";

    for (srcId = 0; srcId < geometry->nrel; srcId++)
    {
        show_information();
        
        initialization();
        forward_solver();
        set_seismogram();
    }
}

void Inversion::show_information()
{
    auto clear = system("clear");

    padding = (WIDTH - title.length() + 8) / 2;

    std::string line(WIDTH, '-');

    std::cout << line << '\n';
    std::cout << std::string(padding, ' ') << title << '\n';
    std::cout << line << "\n\n";
    
    std::cout << "Model dimensions: (z = " << (nz - 1)*dh << 
                                  ", x = " << (nx - 1)*dh <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] << ") m\n\n";

    std::cout << line << "\n";
    std::cout << stage_info << "\n";
    std::cout << line << "\n\n";

    if (!ABC)
    {
        if (iteration >= max_iteration)
        {
            std::cout << "Checking final residuals\n\n";
        }
        else
        {    
            std::cout << "Computing iteration " << iteration + 1 << " of " << max_iteration << "\n\n";

            if (iteration > 0) std::cout << "Previous residuals: " << residuo.back() << "\n";   
        }
    }
}

void Inversion::check_convergence()
{
    float square_difference = 0.0f;

    for (int index = 0; index < nt*geometry->nTraces; index++)
    {
        seismic_data[index] = obs_data[index] - seismic_data[index];
        square_difference += seismic_data[index]*seismic_data[index];
    }

    residuo.push_back(sqrtf(square_difference));

    if ((iteration >= max_iteration))
    {
        std::cout << "Final residuals: "<< residuo.back() <<"\n";
        converged = true;
    }
    else
    {
        iteration += 1;
        converged = false;
    }
}

void Inversion::compute_gradient()
{
    ABC = false;

    nb = rbc_nb;
    nxx = rbc_nxx;
    nzz = rbc_nzz;
    matsize = rbc_matsize;

    nBlocks = (int)((matsize + NTHREADS - 1) / NTHREADS);

    cudaMemset(d_gradient, 0.0f, rbc_matsize*sizeof(float));

    for (srcId = 0; srcId < geometry->nrel; srcId++)
    {
        forward_propagation();
        backward_propagation();
    }
}

void Inversion::forward_propagation()
{
    stage_info = "Calculating gradient of the objective function ---> Forward propagation.";

    show_information();

    initialization();
    rbc_forward_solver();
}

void Inversion::rbc_forward_solver()
{
    cudaMemset(d_Ps, 0.0f, rbc_matsize*sizeof(float));
    cudaMemset(d_Psold, 0.0f, rbc_matsize*sizeof(float));

    set_random_boundary(d_rbc_Vp, rbc_ratio, rbc_varVp);

    for (int tId = 0; tId < tlag + nt; tId++)
    {
        compute_pressure<<<nBlocks,NTHREADS>>>(d_rbc_Vp, d_Ps, d_Psold, d_wavelet, d_b1d, d_b2d, sIdx, sIdz, tId, nt, nb, nxx, nzz, dh, dt, ABC);

        std::swap(d_Ps, d_Psold);
    }
}

void Inversion::backward_propagation()
{
    stage_info = "Calculating gradient of the objective function ---> Backward propagation.";

    show_information();

    initialization();
    
    set_seismic_source();

    cudaMemset(d_Pr, 0.0f, rbc_matsize*sizeof(float));
    cudaMemset(d_Prold, 0.0f, rbc_matsize*sizeof(float));

    for (int tId = 0; tId < nt + tlag; tId++)
    {
        FWI<<<nBlocks,NTHREADS>>>(d_Ps, d_Psold, d_Pr, d_Prold, d_rbc_Vp, d_seismogram, d_gradient, d_sumPs, d_rIdx, d_rIdz, geometry->spread, tId, tlag, nxx, nzz, nt, dh, dt);
    
        std::swap(d_Ps, d_Psold);
        std::swap(d_Pr, d_Prold);
    }
}

void Inversion::set_seismic_source()
{
    for (int timeId = 0; timeId < nt; timeId++)
        for (int spreadId = 0; spreadId < geometry->spread; spreadId++)
            seismogram[timeId + spreadId*nt] = seismic_data[timeId + spreadId*nt + srcId*geometry->spread*nt];     

    cudaMemcpy(d_seismogram, seismogram, nt*geometry->spread*sizeof(float), cudaMemcpyHostToDevice);
}

void Inversion::optimization()
{
    stage_info = "Optimizing problem with ";

    cudaMemcpy(partial, d_gradient, rbc_matsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, gradient);

    cudaMemcpy(partial, d_sumPs, rbc_matsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, sumPs);

    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        int i = (int)(index % nz);
        int j = (int)(index / nz);

        gradient[index] /= sumPs[index];

        A1[index] = beta1*A1[index] + (1.0f - beta1)*gradient[index];
        A2[index] = beta2*A2[index] + (1.0f - beta2)*gradient[index]*gradient[index];
        
        float A1_hat = A1[index] / (1.0f - beta1);
        float A2_hat = A2[index] / (1.0f - beta2);     

        float m = 1.0f / Vp[(i + abc_nb) + (j + abc_nb)*abc_nzz];

        vp[index] = 1.0f /(m - alpha*A1_hat/(sqrtf(A2_hat) + 1e-8f));
    }
}

void Inversion::update_model()
{
    stage_info = "Updating model ...";

    expand_boundary(vp, h_rbc_Vp);
    cudaMemcpy(d_rbc_Vp, h_rbc_Vp, rbc_matsize*sizeof(float), cudaMemcpyHostToDevice);

    nb = abc_nb;
    nxx = abc_nxx;
    nzz = abc_nzz;
    matsize = abc_matsize;

    expand_boundary(vp, Vp);
    cudaMemcpy(d_Vp, Vp, abc_matsize*sizeof(float), cudaMemcpyHostToDevice);
}

void Inversion::export_convergence()
{
    std::string residuo_path = residuo_folder + "convergence_" + std::to_string(iteration) + "_iterations.txt"; 

    std::ofstream resFile(residuo_path, std::ios::out);
    
    for (int r = 0; r < residuo.size(); r++) 
        resFile << residuo[r] << "\n";

    resFile.close();

    std::cout << "Text file \033[34m" << residuo_path << "\033[0;0m was successfully written." << std::endl;
}

void Inversion::export_final_model()
{
    model_file = output_folder + "model_" + std::to_string(int(fmax)) + "Hz_" + std::to_string(nz) + "x" + std::to_string(nx) + ".bin";

    nb = rbc_nb;
    nxx = rbc_nxx;
    nzz = rbc_nzz;
    matsize = rbc_matsize;

    cudaMemcpy(partial, d_rbc_Vp, matsize*sizeof(float), cudaMemcpyDeviceToHost);    
    reduce_boundary(partial, vp);
    export_binary_float(model_file, vp, nPoints);
}

__global__ void FWI(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * seismogram, float * gradient, float * sumPs, int * rIdx, int * rIdz, int spread, int tId, int tlag, int nxx, int nzz, int nt, float dh, float dt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    if ((index == 0) && (tId < nt))
        for (int rId = 0; rId < spread; rId++)
            Pr[rIdz[rId] + rIdx[rId]*nzz] += seismogram[(nt-tId-1) + rId*nt] / (dh*dh); 

    if((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4)) 
    {
        float d2Ps_dx2 = (- 9.0f*(Psold[i + (j-4)*nzz] + Psold[i + (j+4)*nzz])
                      +   128.0f*(Psold[i + (j-3)*nzz] + Psold[i + (j+3)*nzz])
                      -  1008.0f*(Psold[i + (j-2)*nzz] + Psold[i + (j+2)*nzz])
                      +  8064.0f*(Psold[i + (j+1)*nzz] + Psold[i + (j-1)*nzz])
                      - 14350.0f*(Psold[i + j*nzz]))/(5040.0f*dh*dh);

        float d2Ps_dz2 = (- 9.0f*(Psold[(i-4) + j*nzz] + Psold[(i+4) + j*nzz])
                      +   128.0f*(Psold[(i-3) + j*nzz] + Psold[(i+3) + j*nzz])
                      -  1008.0f*(Psold[(i-2) + j*nzz] + Psold[(i+2) + j*nzz])
                      +  8064.0f*(Psold[(i-1) + j*nzz] + Psold[(i+1) + j*nzz])
                      - 14350.0f*(Psold[i + j*nzz]))/(5040.0f*dh*dh);
        
        float d2Pr_dx2 = (- 9.0f*(Pr[i + (j-4)*nzz] + Pr[i + (j+4)*nzz])
                      +   128.0f*(Pr[i + (j-3)*nzz] + Pr[i + (j+3)*nzz])
                      -  1008.0f*(Pr[i + (j-2)*nzz] + Pr[i + (j+2)*nzz])
                      +  8064.0f*(Pr[i + (j+1)*nzz] + Pr[i + (j-1)*nzz])
                      - 14350.0f*(Pr[i + j*nzz]))/(5040.0f*dh*dh);

        float d2Pr_dz2 = (- 9.0f*(Pr[(i-4) + j*nzz] + Pr[(i+4) + j*nzz])
                      +   128.0f*(Pr[(i-3) + j*nzz] + Pr[(i+3) + j*nzz])
                      -  1008.0f*(Pr[(i-2) + j*nzz] + Pr[(i+2) + j*nzz])
                      +  8064.0f*(Pr[(i-1) + j*nzz] + Pr[(i+1) + j*nzz])
                      - 14350.0f*(Pr[i + j*nzz]))/(5040.0f*dh*dh);
        
        Ps[index] = dt*dt*Vp[index]*Vp[index]*(d2Ps_dx2 + d2Ps_dz2) + 2.0f*Psold[index] - Ps[index];    

        Prold[index] = dt*dt*Vp[index]*Vp[index]*(d2Pr_dx2 + d2Pr_dz2) + 2.0f*Pr[index] - Prold[index];

        gradient[index] += dt*Pr[index]*(d2Ps_dx2 + d2Ps_dz2)*Vp[index]*Vp[index];   

        sumPs[index] += Ps[index]*Ps[index];
    }
}
