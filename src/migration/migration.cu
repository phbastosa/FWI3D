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

void Migration::show_information()
{
    auto clear = system("clear");

    padding = (WIDTH - title.length() + 8) / 2;

    std::string line(WIDTH, '-');

    std::cout << line << "\n";
    std::cout << std::string(padding, ' ') << title << '\n';
    std::cout << line << "\n\n";

    std::cout << "Model dimensions: (z = " << (nz - 1)*dh << 
                                  ", x = " << (nx - 1)*dh <<
                                  ", y = " << (ny - 1)*dh <<") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] <<
                                       ", y = " << geometry->ysrc[geometry->sInd[srcId]] << ") m\n\n";

    std::cout << line << "\n";
    std::cout << stage_info << std::endl;
    std::cout << line << "\n";                                                                          
}

void Migration::forward_propagation()
{   
    stage_info = "Forward propagation";

    show_information();

    set_random_boundary(d_Vp, rbc_ratio, rbc_varVp);
    
    initialization();
    forward_solver();
}

void Migration::backward_propagation()
{
    stage_info = "Backward propagation";

    show_information();

    initialization();
    set_seismic_source();

    cudaMemset(d_Pr, 0.0f, volsize*sizeof(float));
    cudaMemset(d_Prold, 0.0f, volsize*sizeof(float));

    for (int tId = 0; tId < nt + tlag; tId++)
    {
        RTM<<<nBlocks, NTHREADS>>>(d_P, d_Pold, d_Pr, d_Prold, d_Vp, d_seismogram, d_image, d_sumPs, d_rIdx, d_rIdy, d_rIdz, geometry->spread, tId, tlag, nxx, nyy, nzz, nt, dh, dt);
    
        std::swap(d_P, d_Pold);
        std::swap(d_Pr, d_Prold);
    }
}

void Migration::set_seismic_source()
{
    std::string data_file = data_folder + "seismogram_nt" + std::to_string(nt) + "_nr" + std::to_string(geometry->spread) + "_" + std::to_string(int(1e6f*dt)) + "us_shot_" + std::to_string(srcId+1) + ".bin";
    import_binary_float(data_file, seismogram, nt*geometry->spread);
    cudaMemcpy(d_seismogram, seismogram, nt*geometry->spread*sizeof(float), cudaMemcpyHostToDevice);
}

void Migration::export_seismic()
{
    cudaMemcpy(partial, d_image, volsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, image);

    cudaMemcpy(partial, d_sumPs, volsize*sizeof(float), cudaMemcpyDeviceToHost);
    reduce_boundary(partial, sumPs);

    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
        image[index] = image[index] / sumPs[index];

    # pragma omp parallel for    
    for (int index = 0; index < nPoints; index++)
    {
        int k = (int) (index / (nx*nz));         
        int j = (int) (index - k*nx*nz) / nz;   
        int i = (int) (index - j*nz - k*nx*nz);      

        if((i > 0) && (i < nz-1) && (j > 0) && (j < nx-1) && (k > 0) && (k < ny-1)) 
        {
            float d2I_dx2 = (image[i + (j-1)*nz + k*nx*nz] - 2.0f*image[index] + image[i + (j+1)*nz + k*nx*nz]) / (dh * dh);
            float d2I_dy2 = (image[i + j*nz + (k-1)*nx*nz] - 2.0f*image[index] + image[i + j*nz + (k+1)*nx*nz]) / (dh * dh);
            float d2I_dz2 = (image[(i-1) + j*nz + k*nx*nz] - 2.0f*image[index] + image[(i+1) + j*nz + k*nx*nz]) / (dh * dh);

            sumPs[index] = d2I_dx2 + d2I_dy2 + d2I_dz2;
        }
        else sumPs[index] = 0.0f;
    }

    std::string output_file = output_folder + "RTM_section_" + std::to_string(nz) + "x" + std::to_string(nx) + "x" + std::to_string(ny) + ".bin";
    export_binary_float(output_file, sumPs, nPoints);
}

__global__ void RTM(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * seismogram, float * image, float * sumPs, int * rIdx, int * rIdy, int * rIdz, int spread, int tId, int tlag, int nxx, int nyy, int nzz, int nt, float dh, float dt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx*nzz);    

    if ((index == 0) && (tId < nt))
        for (int rId = 0; rId < spread; rId++)
            Pr[rIdz[rId] + rIdx[rId]*nzz + rIdy[rId]*nxx*nzz] += seismogram[(nt-tId-1) + rId*nt] / (dh*dh); 
    
    if((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4) && (k > 3) && (k < nyy-4)) 
    {
        float d2Ps_dx2 = (- 9.0f*(Psold[i + (j-4)*nzz + k*nxx*nzz] + Psold[i + (j+4)*nzz + k*nxx*nzz])
                      +   128.0f*(Psold[i + (j-3)*nzz + k*nxx*nzz] + Psold[i + (j+3)*nzz + k*nxx*nzz])
                      -  1008.0f*(Psold[i + (j-2)*nzz + k*nxx*nzz] + Psold[i + (j+2)*nzz + k*nxx*nzz])
                      +  8064.0f*(Psold[i + (j-1)*nzz + k*nxx*nzz] + Psold[i + (j+1)*nzz + k*nxx*nzz])
                      - 14350.0f*(Psold[i + j*nzz + k*nxx*nzz]))/(5040.0f*dh*dh);

        float d2Ps_dy2 = (- 9.0f*(Psold[i + j*nzz + (k-4)*nxx*nzz] + Psold[i + j*nzz + (k+4)*nxx*nzz])
                      +   128.0f*(Psold[i + j*nzz + (k-3)*nxx*nzz] + Psold[i + j*nzz + (k+3)*nxx*nzz])
                      -  1008.0f*(Psold[i + j*nzz + (k-2)*nxx*nzz] + Psold[i + j*nzz + (k+2)*nxx*nzz])
                      +  8064.0f*(Psold[i + j*nzz + (k-1)*nxx*nzz] + Psold[i + j*nzz + (k+1)*nxx*nzz])
                      - 14350.0f*(Psold[i + j*nzz + k*nxx*nzz]))/(5040.0f*dh*dh);

        float d2Ps_dz2 = (- 9.0f*(Psold[(i-4) + j*nzz + k*nxx*nzz] + Psold[(i+4) + j*nzz + k*nxx*nzz])
                      +   128.0f*(Psold[(i-3) + j*nzz + k*nxx*nzz] + Psold[(i+3) + j*nzz + k*nxx*nzz])
                      -  1008.0f*(Psold[(i-2) + j*nzz + k*nxx*nzz] + Psold[(i+2) + j*nzz + k*nxx*nzz])
                      +  8064.0f*(Psold[(i-1) + j*nzz + k*nxx*nzz] + Psold[(i+1) + j*nzz + k*nxx*nzz])
                      - 14350.0f*(Psold[i + j*nzz + k*nxx*nzz]))/(5040.0f*dh*dh);
        
        float d2Pr_dx2 = (- 9.0f*(Pr[i + (j-4)*nzz + k*nxx*nzz] + Pr[i + (j+4)*nzz + k*nxx*nzz])
                      +   128.0f*(Pr[i + (j-3)*nzz + k*nxx*nzz] + Pr[i + (j+3)*nzz + k*nxx*nzz])
                      -  1008.0f*(Pr[i + (j-2)*nzz + k*nxx*nzz] + Pr[i + (j+2)*nzz + k*nxx*nzz])
                      +  8064.0f*(Pr[i + (j-1)*nzz + k*nxx*nzz] + Pr[i + (j+1)*nzz + k*nxx*nzz])
                      - 14350.0f*(Pr[i + j*nzz + k*nxx*nzz]))/(5040.0f*dh*dh);

        float d2Pr_dy2 = (- 9.0f*(Pr[i + j*nzz + (k-4)*nxx*nzz] + Pr[i + j*nzz + (k+4)*nxx*nzz])
                      +   128.0f*(Pr[i + j*nzz + (k-3)*nxx*nzz] + Pr[i + j*nzz + (k+3)*nxx*nzz])
                      -  1008.0f*(Pr[i + j*nzz + (k-2)*nxx*nzz] + Pr[i + j*nzz + (k+2)*nxx*nzz])
                      +  8064.0f*(Pr[i + j*nzz + (k-1)*nxx*nzz] + Pr[i + j*nzz + (k+1)*nxx*nzz])
                      - 14350.0f*(Pr[i + j*nzz + k*nxx*nzz]))/(5040.0f*dh*dh);

        float d2Pr_dz2 = (- 9.0f*(Pr[(i-4) + j*nzz + k*nxx*nzz] + Pr[(i+4) + j*nzz + k*nxx*nzz])
                      +   128.0f*(Pr[(i-3) + j*nzz + k*nxx*nzz] + Pr[(i+3) + j*nzz + k*nxx*nzz])
                      -  1008.0f*(Pr[(i-2) + j*nzz + k*nxx*nzz] + Pr[(i+2) + j*nzz + k*nxx*nzz])
                      +  8064.0f*(Pr[(i-1) + j*nzz + k*nxx*nzz] + Pr[(i+1) + j*nzz + k*nxx*nzz])
                      - 14350.0f*(Pr[i + j*nzz + k*nxx*nzz]))/(5040.0f*dh*dh);
        
        Ps[index] = dt*dt*Vp[index]*Vp[index]*(d2Ps_dx2 + d2Ps_dy2 + d2Ps_dz2) + 2.0f*Psold[index] - Ps[index];    

        Prold[index] = dt*dt*Vp[index]*Vp[index]*(d2Pr_dx2 + d2Pr_dy2 + d2Pr_dz2) + 2.0f*Pr[index] - Prold[index];
    
        sumPs[index] += Ps[index]*Ps[index]; 
        image[index] += Ps[index]*Pr[index];
    }
}