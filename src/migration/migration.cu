# include "migration.cuh"

void Migration::set_parameters()
{    
    title = "\033[34mReverse Time Migration\033[0;0m";

    set_main_parameters();

    rbc_ratio = std::stof(catch_parameter("mig_rbc_ratio", parameters)); 
    rbc_varVp = std::stof(catch_parameter("mig_rbc_varVp", parameters)); 
    rbc_length = std::stof(catch_parameter("mig_rbc_length", parameters));

    nb = (int)(rbc_length / dh) + 1;

    set_wavelet();
    set_geometry();    
    set_seismograms();
    set_properties();
    set_coordinates();

    input_folder = catch_parameter("mig_input_folder", parameters);
    input_prefix = catch_parameter("mig_input_prefix", parameters);    
    output_folder = catch_parameter("mig_output_folder", parameters);

    image = new float[nPoints]();
    sumPs = new float[nPoints]();
    partial = new float[volsize]();

    cudaMalloc((void**)&(d_Pr), volsize*sizeof(float));
    cudaMalloc((void**)&(d_Prold), volsize*sizeof(float));
    cudaMalloc((void**)&(d_image), volsize*sizeof(float));
    cudaMalloc((void**)&(d_sumPs), volsize*sizeof(float));

    cudaMemset(d_image, 0.0f, volsize*sizeof(float));
    cudaMemset(d_sumPs, 0.0f, volsize*sizeof(float));
}

void Migration::show_mig_info()
{
    show_information();

    std::string line(WIDTH, '-');

    std::cout << line << "\n";
    std::cout << stage_info << std::endl;
    std::cout << line << "\n";                                                                        
}

void Migration::forward_propagation()
{   
    stage_info = "Forward propagation";

    show_mig_info();

    set_random_boundary(d_Vp, rbc_ratio, rbc_varVp);
    
    initialization();
    forward_solver();
}

void Migration::backward_propagation()
{
    stage_info = "Backward propagation";

    show_mig_info();

    set_seismic_source();

    cudaMemset(d_Pr, 0.0f, volsize*sizeof(float));
    cudaMemset(d_Prold, 0.0f, volsize*sizeof(float));

    for (int tId = 0; tId < nt + tlag; tId++)
    {
        inject_seismogram<<<sBlocks, NTHREADS>>>(d_Pr, d_rIdx, d_rIdy, d_rIdz, d_seismogram, geometry->nrec, tId, nt, nxx, nzz, idh3);

        cross_correlation<<<nBlocks, NTHREADS>>>(d_P, d_Pold, d_Pr, d_Prold, d_Vp, d_image, d_sumPs, nxx, nyy, nzz, nt, idh2, dt);
    
        std::swap(d_P, d_Pold);
        std::swap(d_Pr, d_Prold);
    }
}

void Migration::set_seismic_source()
{
    std::string data_file = input_folder + input_prefix + std::to_string(srcId+1) + ".bin";
    import_binary_float(data_file, seismogram, nt*geometry->nrec);
    cudaMemcpy(d_seismogram, seismogram, nt*geometry->nrec*sizeof(float), cudaMemcpyHostToDevice);
}

void Migration::export_seismic()
{
    cudaMemcpy(partial, d_image, volsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, image);

    cudaMemcpy(partial, d_sumPs, volsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, sumPs);

    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
        sumPs[index] = image[index] / sumPs[index];

    # pragma omp parallel for    
    for (int index = 0; index < nPoints; index++)
    {
        int k = (int) (index / (nx*nz));         
        int j = (int) (index - k*nx*nz) / nz;   
        int i = (int) (index - j*nz - k*nx*nz);      

        image[index] = 0.0f;

        if((i > 0) && (i < nz-1)) 
            image[index] = -1.0f*(sumPs[index-1] - 2.0f*sumPs[index] + sumPs[index+1]) * idh2;    
    }

    std::string output_file = output_folder + "RTM_section_" + std::to_string(int(fmax)) + "Hz_" + std::to_string(nz) + "x" + std::to_string(nx) + "x" + std::to_string(ny) + "_" + std::to_string((int)(dh)) + "m.bin";
    export_binary_float(output_file, image, nPoints);
}

__global__ void inject_seismogram(float * __restrict__ Pr, const int * __restrict__ rIdx, const int * __restrict__ rIdy, 
                                  const int * __restrict__ rIdz, const float * __restrict__ seismogram, int spread, int tId, 
                                  int nt, int nxx, int nzz, float idh3)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < spread) && (tId < nt))
    {
        int sId = (nt-tId-1) + index*nt;
        int wId = rIdz[index] + rIdx[index]*nzz + rIdy[index]*nxx*nzz;

        atomicAdd(&Pr[wId], idh3*seismogram[sId]);
    }
}

__global__ void cross_correlation(float * __restrict__ Ps, const float * __restrict__ Psold, const float * __restrict__ Pr, 
                                  float * __restrict__ Prold, const float * __restrict__ Vp, float * __restrict__ image, 
                                  float * __restrict__ sumPs, int nxx, int nyy, int nzz, int nt, float idh2, float dt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int nxx_nzz = nxx*nzz;

    int k = (int) (index / nxx_nzz);         
    int j = (int) (index - k*nxx_nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx_nzz); 

    const int base_j = j*nzz;
    const int base_k = k*nxx_nzz;

    const int jm4 = base_j - 4*nzz, jm3 = base_j - 3*nzz, jm2 = base_j - 2*nzz, jm1 = base_j - nzz;
    const int jp4 = base_j + 4*nzz, jp3 = base_j + 3*nzz, jp2 = base_j + 2*nzz, jp1 = base_j + nzz;

    const int km4 = base_k - 4*nxx_nzz, km3 = base_k - 3*nxx_nzz, km2 = base_k - 2*nxx_nzz, km1 = base_k - nxx_nzz;     
    const int kp4 = base_k + 4*nxx_nzz, kp3 = base_k + 3*nxx_nzz, kp2 = base_k + 2*nxx_nzz, kp1 = base_k + nxx_nzz;     
    
    if((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4) && (k > 3) && (k < nyy-4)) 
    {
        const float vp2 = Vp[index]*Vp[index];

        float d2Ps_dx2 = (-FDM1*(Psold[i + jm4 + base_k] + Psold[i + jp4 + base_k])
                          +FDM2*(Psold[i + jm3 + base_k] + Psold[i + jp3 + base_k])
                          -FDM3*(Psold[i + jm2 + base_k] + Psold[i + jp2 + base_k])
                          +FDM4*(Psold[i + jm1 + base_k] + Psold[i + jp1 + base_k])
                          -FDM5*(Psold[i + base_j + base_k]))*idh2;

        float d2Ps_dy2 = (-FDM1*(Psold[i + base_j + km4] + Psold[i + base_j + kp4])
                          +FDM2*(Psold[i + base_j + km3] + Psold[i + base_j + kp3])
                          -FDM3*(Psold[i + base_j + km2] + Psold[i + base_j + kp2])
                          +FDM4*(Psold[i + base_j + km1] + Psold[i + base_j + kp1])
                          -FDM5*(Psold[i + base_j + base_k]))*idh2;

        float d2Ps_dz2 = (-FDM1*(Psold[(i-4) + base_j + base_k] + Psold[(i+4) + base_j + base_k])
                          +FDM2*(Psold[(i-3) + base_j + base_k] + Psold[(i+3) + base_j + base_k])
                          -FDM3*(Psold[(i-2) + base_j + base_k] + Psold[(i+2) + base_j + base_k])
                          +FDM4*(Psold[(i-1) + base_j + base_k] + Psold[(i+1) + base_j + base_k])
                          -FDM5*(Psold[i + base_j + base_k]))*idh2;
        
        float d2Pr_dx2 = (-FDM1*(Pr[i + jm4 + base_k] + Pr[i + jp4 + base_k])
                          +FDM2*(Pr[i + jm3 + base_k] + Pr[i + jp3 + base_k])
                          -FDM3*(Pr[i + jm2 + base_k] + Pr[i + jp2 + base_k])
                          +FDM4*(Pr[i + jm1 + base_k] + Pr[i + jp1 + base_k])
                          -FDM5*(Pr[i + base_j + base_k]))*idh2;

        float d2Pr_dy2 = (-FDM1*(Pr[i + base_j + km4] + Pr[i + base_j + kp4])
                          +FDM2*(Pr[i + base_j + km3] + Pr[i + base_j + kp3])
                          -FDM3*(Pr[i + base_j + km2] + Pr[i + base_j + kp2])
                          +FDM4*(Pr[i + base_j + km1] + Pr[i + base_j + kp1])
                          -FDM5*(Pr[i + base_j + base_k]))*idh2;

        float d2Pr_dz2 = (-FDM1*(Pr[(i-4) + base_j + base_k] + Pr[(i+4) + base_j + base_k])
                          +FDM2*(Pr[(i-3) + base_j + base_k] + Pr[(i+3) + base_j + base_k])
                          -FDM3*(Pr[(i-2) + base_j + base_k] + Pr[(i+2) + base_j + base_k])
                          +FDM4*(Pr[(i-1) + base_j + base_k] + Pr[(i+1) + base_j + base_k])
                          -FDM5*(Pr[i + base_j + base_k]))*idh2;
        
        Ps[index] = dt*dt*vp2*(d2Ps_dx2 + d2Ps_dy2 + d2Ps_dz2) + 2.0f*Psold[index] - Ps[index];    

        Prold[index] = dt*dt*vp2*(d2Pr_dx2 + d2Pr_dy2 + d2Pr_dz2) + 2.0f*Pr[index] - Prold[index];
    
        atomicAdd(&sumPs[index], Ps[index]*Ps[index]);
        atomicAdd(&image[index], Ps[index]*Pr[index]);
    }
}