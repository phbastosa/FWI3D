# include "inversion.cuh"

void Inversion::set_parameters()
{
    title = "\033[34mFull Waveform Inversion\033[0;0m";
    
    set_main_parameters();

    set_abc_dampers();

    set_wavelet();
    set_geometry();    
    set_properties();
    set_seismograms();
    
    set_ABC_dimension();
    set_RBC_dimension();
    get_RBC_dimension();
    set_coordinates();

    zmask = std::stof(catch_parameter("depth_mask", parameters));

    data_folder = catch_parameter("inv_dcal_folder", parameters);
    input_folder = catch_parameter("inv_input_folder", parameters);
    input_prefix = catch_parameter("inv_input_prefix", parameters);
    output_folder = catch_parameter("inv_output_folder", parameters);
    residuo_folder = catch_parameter("inv_residuo_folder", parameters);

    max_iteration = std::stoi(catch_parameter("max_iteration", parameters));

    iteration = 0;

    model = new float[nPoints]();

    partial1 = new float[nPoints]();
    partial2 = new float[rbc_volsize]();

    gradient = new float[nPoints]();
    obs_data = new float[nt*geometry->nrec]();

    set_initial_model();

    cudaMalloc((void**)&(d_Vp_rbc), rbc_volsize*sizeof(float));
    
    cudaMalloc((void**)&(d_Ps_rbc), rbc_volsize*sizeof(float));
    cudaMalloc((void**)&(d_Pr_rbc), rbc_volsize*sizeof(float));
    
    cudaMalloc((void**)&(d_Ps_old_rbc), rbc_volsize*sizeof(float));
    cudaMalloc((void**)&(d_Pr_old_rbc), rbc_volsize*sizeof(float));
    
    cudaMalloc((void**)&(d_sumPs_rbc), rbc_volsize*sizeof(float));
    cudaMalloc((void**)&(d_gradient_rbc), rbc_volsize*sizeof(float));
}

void Inversion::set_ABC_dimension()
{
    abc_nb = (int)(abc_length / dh) + 1;

    abc_nxx = nx + 2*abc_nb;
    abc_nyy = ny + 2*abc_nb;
    abc_nzz = nz + 2*abc_nb;
    abc_volsize = abc_nxx*abc_nyy*abc_nzz;
}

void Inversion::get_ABC_dimension()
{
    ABC = true;

    nb = abc_nb;
    nxx = abc_nxx;
    nyy = abc_nyy;
    nzz = abc_nzz;
    volsize = abc_volsize;

    nBlocks = (int)((abc_volsize + NTHREADS - 1) / NTHREADS);
}

void Inversion::set_RBC_dimension()
{
    rbc_ratio = std::stof(catch_parameter("inv_rbc_ratio", parameters)); 
    rbc_varVp = std::stof(catch_parameter("inv_rbc_varVp", parameters)); 
    rbc_length = std::stof(catch_parameter("inv_rbc_length", parameters));

    rbc_nb = (int)(rbc_length / dh) + 1;

    rbc_nxx = nx + 2*rbc_nb;
    rbc_nyy = ny + 2*rbc_nb;
    rbc_nzz = nz + 2*rbc_nb;
    rbc_volsize = rbc_nxx*rbc_nyy*rbc_nzz;
}

void Inversion::get_RBC_dimension()
{
    ABC = false;

    nb = rbc_nb;
    nxx = rbc_nxx;
    nyy = rbc_nyy;
    nzz = rbc_nzz;
    volsize = rbc_volsize;

    nBlocks = (int)((rbc_volsize + NTHREADS - 1) / NTHREADS);
}

void Inversion::set_initial_model()
{
    get_ABC_dimension();
    reduce_boundary(Vp, model);
    
    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
        model[index] = 1.0f / (model[index]*model[index]);
}

void Inversion::compute_gradient()
{
    sum_res = 0;

    cudaMemset(d_gradient_rbc, 0.0f, rbc_volsize*sizeof(float));

    if (iteration == max_iteration) ++iteration;
    
    for (srcId = 0; srcId < geometry->nsrc; srcId++)
    {
        set_obs_data();
        get_cal_data();
        
        set_adjoint_source();

        if (iteration <= max_iteration)
        {
            update_RBC();

            initialization();
            forward_propagation();
            backward_propagation();
        }
    }

    if (iteration == max_iteration) --iteration;
    
    cudaMemcpy(partial2, d_gradient_rbc, rbc_volsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial2, gradient);

    cudaMemcpy(partial2, d_sumPs_rbc, rbc_volsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial2, partial1);

    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
        partial1[index] = gradient[index] / partial1[index];

    float gmax = -1e9f;

    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        int i = (int)(index % nz);

        gradient[index] = 0.0f;    

        if((i > (int)(zmask/dh)) && (i < nz-1)) 
            gradient[index] = -1.0f*(partial1[index-1] - 2.0f*partial1[index] + partial1[index+1]) * idh2;
    
        gmax = gmax < fabsf(gradient[index]) ? fabsf(gradient[index]) : gmax; 
    }

    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
        gradient[index] *= 1.0f / gmax;
}

void Inversion::set_obs_data()
{
    std::string data_file = input_folder + input_prefix + std::to_string(srcId+1) + ".bin";
    import_binary_float(data_file, obs_data, nt*geometry->nrec);
}

void Inversion::get_cal_data()
{
    stage_info = "Computing sinthetic data";

    show_inv_info();

    get_ABC_dimension();

    initialization();
    forward_solver();
}

void Inversion::set_adjoint_source()
{
    cudaMemcpy(seismogram, d_seismogram, nt*geometry->nrec*sizeof(float), cudaMemcpyDeviceToHost);

    for (int index = 0; index < nt*geometry->nrec; index++)
    {
        seismogram[index] = obs_data[index] - seismogram[index];
        sum_res += seismogram[index]*seismogram[index];
    }    

    cudaMemcpy(d_seismogram, seismogram, nt*geometry->nrec*sizeof(float), cudaMemcpyHostToDevice);
}


void Inversion::update_RBC()
{
    cudaMemcpy(Vp, d_Vp, abc_volsize*sizeof(float), cudaMemcpyDeviceToHost);

    get_ABC_dimension();
    reduce_boundary(Vp, partial1);
    get_RBC_dimension();
    expand_boundary(partial1, partial2);

    cudaMemcpy(d_Vp_rbc, partial2, rbc_volsize*sizeof(float), cudaMemcpyHostToDevice);

    set_random_boundary(d_Vp_rbc, rbc_ratio, rbc_varVp);
}

void Inversion::forward_propagation()
{
    stage_info = "Wavefield reconstruction: forward propagation";

    show_inv_info();

    cudaMemset(d_Ps_rbc, 0.0f, rbc_volsize*sizeof(float));
    cudaMemset(d_Ps_old_rbc, 0.0f, rbc_volsize*sizeof(float));

    for (int tId = 0; tId < tlag + nt; tId++)
    {
        compute_pressure<<<nBlocks,NTHREADS>>>(d_Vp_rbc, d_Ps_rbc, d_Ps_old_rbc, d_wavelet, d_b1d, d_b2d, d_b3d, sIdx, sIdy, sIdz, tId, nt, nb, nxx, nyy, nzz, idh2, idh3, dt, ABC);

        std::swap(d_Ps_rbc, d_Ps_old_rbc);
    }    
}

void Inversion::backward_propagation()
{
    stage_info = "Wavefield reconstruction: backward propagation";

    show_inv_info();
    
    cudaMemset(d_Pr_rbc, 0.0f, rbc_volsize*sizeof(float));
    cudaMemset(d_Pr_old_rbc, 0.0f, rbc_volsize*sizeof(float));

    for (int tId = 0; tId < nt + tlag; tId++)
    {
        inject_residuo<<<sBlocks,NTHREADS>>>(d_Pr_rbc, d_rIdx, d_rIdy, d_rIdz, d_seismogram, geometry->nrec, tId, nt, nxx, nzz, idh3);

        build_gradient<<<nBlocks,NTHREADS>>>(d_Ps_rbc, d_Ps_old_rbc, d_Pr_rbc, d_Pr_old_rbc, d_Vp_rbc, d_gradient_rbc, d_sumPs_rbc, nxx, nyy, nzz, nt, dt, idh2);
    
        std::swap(d_Ps_rbc, d_Ps_old_rbc);
        std::swap(d_Pr_rbc, d_Pr_old_rbc);
    }
}

void Inversion::show_inv_info()
{
    show_information();

    std::string line(WIDTH, '-');

    std::cout << line << "\n";
    std::cout << stage_info << "\n";
    std::cout << line << "\n\n";

    if (iteration > max_iteration) 
        std::cout << "-------- Checking final residuo --------\n\n";
    else
    {    
        if (iteration == 0) 
            std::cout << "-------- Computing first residuo --------\n";        
        else
        {
            std::cout << "-------- Computing iteration " << iteration << " of " << max_iteration << " --------\n\n";
            
            std::cout << "Previous residuo: " << residuo.back() << "\n\n";   
        }
    }
}

void Inversion::check_convergence()
{
    ++iteration;
    
    residuo.push_back(sqrtf(sum_res));    

    converged = (iteration > max_iteration) ? true : false;

    if (converged) std::cout << "Final residuo: "<< residuo.back() <<"\n\n";  
}

void Inversion::optimization()
{
    float dm = (1.0f / (vmin*vmin)) - (1.0f / (vmax*vmax));

    float a0 = 0.00f*dm;
    float a1 = 0.02f*dm;
    float a2 = 0.05f*dm;

    float f0 = residuo.back();

    stage_info = "Optimization via parabolic linesearch: first modeling";

    linesearch(a1); float f1 = sqrtf(sum_res);

    stage_info = "Optimization via parabolic linesearch: final modeling";
    
    linesearch(a2); float f2 = sqrtf(sum_res);

    float num = (a1*a1 - a2*a2)*f0 + (a2*a2 - a0*a0)*f1 + (a0*a0 - a1*a1)*f2;    
    float den = (a1 - a2)*f0 + (a2 - a0)*f1 + (a0 - a1)*f2;

    step = 0.5f*(num / den);
}

void Inversion::linesearch(float alpha)
{
    for (int index = 0; index < nPoints; index++)
    {
        partial1[index] = model[index] - alpha*gradient[index];
        partial1[index] = 1.0f / sqrtf(partial1[index]);
    }

    get_ABC_dimension();
    expand_boundary(partial1, Vp);

    cudaMemcpy(d_Vp, Vp, abc_volsize*sizeof(float), cudaMemcpyHostToDevice);

    sum_res = 0.0f;

    for (srcId = 0; srcId < geometry->nsrc; srcId++)    
    {
        set_obs_data();

        show_inv_info();

        initialization();
        forward_solver();
        
        cudaMemcpy(seismogram, d_seismogram, nt*geometry->nrec*sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int index = 0; index < nt*geometry->nrec; index++)
        {    
            seismogram[index] = obs_data[index] - seismogram[index];
            sum_res += seismogram[index]*seismogram[index];
        }
    }
}

void Inversion::update_model()
{
    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        model[index] = model[index] - step*gradient[index];
        partial1[index] = 1.0f / sqrtf(model[index]);
    }

    expand_boundary(partial1, Vp);
    
    std::string model_file = output_folder + "model_FWI_iteration_" + std::to_string(iteration) + "_" + std::to_string(int(fmax)) + "Hz_" + std::to_string(nz) + "x" + std::to_string(nx) + "x" + std::to_string(ny) + "_" + std::to_string((int)(dh)) + "m.bin";
    export_binary_float(model_file, partial1, nPoints);
}

void Inversion::export_convergence()
{
    std::string residuo_path = residuo_folder + "convergence_" + std::to_string(max_iteration) + "_iterations.txt"; 

    std::ofstream resFile(residuo_path, std::ios::out);
    
    for (int r = 0; r <= max_iteration; r++) 
        resFile << residuo[r] << "\n";

    resFile.close();

    std::cout << "Text file \033[34m" << residuo_path << "\033[0;0m was successfully written." << std::endl;
}

void Inversion::export_final_model()
{
    std::string model_file = output_folder + "final_model_FWI_" + std::to_string(int(fmax)) + "Hz_" + std::to_string(nz) + "x" + std::to_string(nx) + "x" + std::to_string(ny) + "_" + std::to_string((int)(dh)) + "m.bin";
    reduce_boundary(Vp, partial1);
    export_binary_float(model_file, partial1, nPoints);
}

__global__ void inject_residuo(float * __restrict__ Pr, const int * __restrict__ rIdx, const int * __restrict__ rIdy, 
                               const int * __restrict__ rIdz, const float * __restrict__ seismogram, int nr, int tId, 
                               int nt, int nxx, int nzz, float idh3)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < nr) && (tId < nt))
    {
        int sId = (nt-tId-1) + index*nt;
        int wId = rIdz[index] + rIdx[index]*nzz + rIdy[index]*nxx*nzz;

        atomicAdd(&Pr[wId], idh3*seismogram[sId]);
    }
}

__global__ void build_gradient(float * __restrict__ Ps, const float * __restrict__ Psold, const float * __restrict__ Pr, 
                               float * __restrict__ Prold, const float * __restrict__ Vp, float * __restrict__ gradient, 
                               float * __restrict__ sumPs, int nxx, int nyy, int nzz, int nt, float dt, float idh2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int nxx_nzz = nxx*nzz;

    int k = (int) (index / nxx_nzz);         
    int j = (int) (index - k*nxx_nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx_nzz); 

    const int bsj = j*nzz;
    const int bsk = k*nxx_nzz;

    const int jm4 = bsj - 4*nzz, jm3 = bsj - 3*nzz, jm2 = bsj - 2*nzz, jm1 = bsj - nzz;
    const int jp4 = bsj + 4*nzz, jp3 = bsj + 3*nzz, jp2 = bsj + 2*nzz, jp1 = bsj + nzz;

    const int km4 = bsk - 4*nxx_nzz, km3 = bsk - 3*nxx_nzz, km2 = bsk - 2*nxx_nzz, km1 = bsk - nxx_nzz;     
    const int kp4 = bsk + 4*nxx_nzz, kp3 = bsk + 3*nxx_nzz, kp2 = bsk + 2*nxx_nzz, kp1 = bsk + nxx_nzz;     
    
    if((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4) && (k > 3) && (k < nyy-4)) 
    {
        const float vp2 = Vp[index]*Vp[index];

        float d2Ps_dx2 = (-FDM1*(Psold[i + jm4 + bsk] + Psold[i + jp4 + bsk])
                          +FDM2*(Psold[i + jm3 + bsk] + Psold[i + jp3 + bsk])
                          -FDM3*(Psold[i + jm2 + bsk] + Psold[i + jp2 + bsk])
                          +FDM4*(Psold[i + jm1 + bsk] + Psold[i + jp1 + bsk])
                          -FDM5*(Psold[i + bsj + bsk]))*idh2;

        float d2Ps_dy2 = (-FDM1*(Psold[i + bsj + km4] + Psold[i + bsj + kp4])
                          +FDM2*(Psold[i + bsj + km3] + Psold[i + bsj + kp3])
                          -FDM3*(Psold[i + bsj + km2] + Psold[i + bsj + kp2])
                          +FDM4*(Psold[i + bsj + km1] + Psold[i + bsj + kp1])
                          -FDM5*(Psold[i + bsj + bsk]))*idh2;

        float d2Ps_dz2 = (-FDM1*(Psold[(i-4) + bsj + bsk] + Psold[(i+4) + bsj + bsk])
                          +FDM2*(Psold[(i-3) + bsj + bsk] + Psold[(i+3) + bsj + bsk])
                          -FDM3*(Psold[(i-2) + bsj + bsk] + Psold[(i+2) + bsj + bsk])
                          +FDM4*(Psold[(i-1) + bsj + bsk] + Psold[(i+1) + bsj + bsk])
                          -FDM5*(Psold[i + bsj + bsk]))*idh2;
        
        float d2Pr_dx2 = (-FDM1*(Pr[i + jm4 + bsk] + Pr[i + jp4 + bsk])
                          +FDM2*(Pr[i + jm3 + bsk] + Pr[i + jp3 + bsk])
                          -FDM3*(Pr[i + jm2 + bsk] + Pr[i + jp2 + bsk])
                          +FDM4*(Pr[i + jm1 + bsk] + Pr[i + jp1 + bsk])
                          -FDM5*(Pr[i + bsj + bsk]))*idh2;

        float d2Pr_dy2 = (-FDM1*(Pr[i + bsj + km4] + Pr[i + bsj + kp4])
                          +FDM2*(Pr[i + bsj + km3] + Pr[i + bsj + kp3])
                          -FDM3*(Pr[i + bsj + km2] + Pr[i + bsj + kp2])
                          +FDM4*(Pr[i + bsj + km1] + Pr[i + bsj + kp1])
                          -FDM5*(Pr[i + bsj + bsk]))*idh2;

        float d2Pr_dz2 = (-FDM1*(Pr[(i-4) + bsj + bsk] + Pr[(i+4) + bsj + bsk])
                          +FDM2*(Pr[(i-3) + bsj + bsk] + Pr[(i+3) + bsj + bsk])
                          -FDM3*(Pr[(i-2) + bsj + bsk] + Pr[(i+2) + bsj + bsk])
                          +FDM4*(Pr[(i-1) + bsj + bsk] + Pr[(i+1) + bsj + bsk])
                          -FDM5*(Pr[i + bsj + bsk]))*idh2;
        
        Ps[index] = dt*dt*vp2*(d2Ps_dx2 + d2Ps_dy2 + d2Ps_dz2) + 2.0f*Psold[index] - Ps[index];    

        Prold[index] = dt*dt*vp2*(d2Pr_dx2 + d2Pr_dy2 + d2Pr_dz2) + 2.0f*Pr[index] - Prold[index];
    
        atomicAdd(&sumPs[index], Ps[index]*Ps[index]);
        atomicAdd(&gradient[index], dt*Pr[index]*(d2Ps_dx2 + d2Ps_dz2)*vp2);
    }
}
