# include "modeling.cuh"

void Modeling::set_parameters()
{
    title = "\033[34mSeismic Modeling\033[0;0m";

    set_main_parameters();

    set_wavelet();
    set_geometry();    
    set_seismograms();
    set_abc_dampers();
    set_properties();
}

void Modeling::set_main_parameters()
{
    nx = std::stoi(catch_parameter("x_samples", parameters));        
    ny = std::stoi(catch_parameter("y_samples", parameters));        
    nz = std::stoi(catch_parameter("z_samples", parameters));        

    dh = std::stof(catch_parameter("model_spacing", parameters));

    nt = std::stoi(catch_parameter("time_samples", parameters));
    dt = std::stof(catch_parameter("time_spacing", parameters));

    fmax = std::stof(catch_parameter("max_frequency", parameters));
    
    data_folder = catch_parameter("mod_output_folder", parameters);
}

void Modeling::set_wavelet()
{
    float * signal_aux = new float[nt]();

    float t0 = 2.0f*sqrtf(M_PI) / fmax;
    float fc = fmax / (3.0f * sqrtf(M_PI));

    tlag = (int)(t0 / dt) + 1;

    for (int n = 0; n < nt; n++)
    {
        float td = n*dt - t0;

        float arg = M_PI*M_PI*M_PI*fc*fc*td*td;

        signal_aux[n] = 1e5f*(1.0f - 2.0f*arg)*expf(-arg);
    }

    cudaMalloc((void**)&(d_wavelet), nt*sizeof(float));

    cudaMemcpy(d_wavelet, signal_aux, nt*sizeof(float), cudaMemcpyHostToDevice);

    delete[] signal_aux;
}

void Modeling::set_geometry()
{
    geometry = new Geometry();
    geometry->parameters = parameters;
    geometry->set_parameters();
    
    cudaMalloc((void**)&(d_rIdx), geometry->spread*sizeof(int));
    cudaMalloc((void**)&(d_rIdy), geometry->spread*sizeof(int));
    cudaMalloc((void**)&(d_rIdz), geometry->spread*sizeof(int));

    cudaMalloc((void**)&(d_sw), NW*NW*NW*sizeof(float));
    cudaMalloc((void**)&(d_rw), NW*NW*NW*geometry->spread*sizeof(float));
}

void Modeling::set_seismograms()
{
    sBlocks = (int)((geometry->spread + NTHREADS - 1) / NTHREADS); 
    
    seismogram = new float[nt*geometry->spread]();

    cudaMalloc((void**)&(d_seismogram), nt*geometry->spread*sizeof(float));
}

void Modeling::set_abc_dampers()
{
    ABC = true;

    abc_length = std::stof(catch_parameter("abc_length", parameters));
    abc_factor = std::stof(catch_parameter("abc_factor", parameters));

    nb = (int)(abc_length / dh) + 4;

    float * damp1D = new float[nb]();
    float * damp2D = new float[nb*nb]();
    float * damp3D = new float[nb*nb*nb]();

    for (int i = 0; i < nb; i++) 
    {
        damp1D[i] = expf(-powf(abc_factor * (nb - i), 2.0f));
    }

    for(int i = 0; i < nb; i++) 
    {
        for (int j = 0; j < nb; j++)
        {   
            damp2D[j + i*nb] += damp1D[i];
            damp2D[i + j*nb] += damp1D[i];
        }
    }

    for (int i  = 0; i < nb; i++)
    {
        for(int j = 0; j < nb; j++)
        {
            for(int k = 0; k < nb; k++)
            {
                damp3D[i + j*nb + k*nb*nb] += damp2D[i + j*nb];
                damp3D[i + j*nb + k*nb*nb] += damp2D[j + k*nb];
                damp3D[i + j*nb + k*nb*nb] += damp2D[i + k*nb];
            }
        }
    }    

    for (int index = 0; index < nb*nb; index++)
        damp2D[index] -= 1.0f;

    for (int index = 0; index < nb*nb*nb; index++)
        damp3D[index] -= 5.0f;    

    cudaMalloc((void**)&(d_b1d), nb*sizeof(float));
    cudaMalloc((void**)&(d_b2d), nb*nb*sizeof(float));
    cudaMalloc((void**)&(d_b3d), nb*nb*nb*sizeof(float));

    cudaMemcpy(d_b1d, damp1D, nb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2d, damp2D, nb*nb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3d, damp3D, nb*nb*nb*sizeof(float), cudaMemcpyHostToDevice);

    delete[] damp1D;
    delete[] damp2D;
    delete[] damp3D;
}

void Modeling::set_properties()
{
    nPoints = nx*ny*nz;

    nxx = nx + 2*nb;
    nyy = ny + 2*nb;
    nzz = nz + 2*nb;

    volsize = nxx*nyy*nzz;

    nBlocks = (int)((volsize + NTHREADS - 1) / NTHREADS);
    
    Vp = new float[volsize]();

    cudaMalloc((void**)&(d_P), volsize*sizeof(float));
    cudaMalloc((void**)&(d_Vp), volsize*sizeof(float));
    cudaMalloc((void**)&(d_Pold), volsize*sizeof(float));

    float * vp = new float[nPoints]();
    std::string vp_file = catch_parameter("model_file", parameters);
    import_binary_float(vp_file, vp, nPoints);

    vmax = 0.0f;
    vmin = 1e9f;

    for (int index = 0; index < nPoints; index++)
    {
        vmax = vmax < vp[index] ? vp[index] : vmax; 
        vmin = vmin > vp[index] ? vp[index] : vmin; 
    }

    expand_boundary(vp, Vp);
    
    cudaMemcpy(d_Vp, Vp, volsize*sizeof(float), cudaMemcpyHostToDevice);

    delete[] vp;
}

void Modeling::expand_boundary(float * input, float * output)
{
    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        int k = (int) (index / (nx*nz));         
        int j = (int) (index - k*nx*nz) / nz;    
        int i = (int) (index - j*nz - k*nx*nz);  

        output[(i + nb) + (j + nb)*nzz + (k + nb)*nxx*nzz] = input[i + j*nz + k*nx*nz];       
    }

    for (int k = nb; k < nyy - nb; k++)
    {   
        for (int j = nb; j < nxx - nb; j++)
        {
            for (int i = 0; i < nb; i++)            
            {
                output[i + j*nzz + k*nxx*nzz] = input[0 + (j - nb)*nz + (k - nb)*nx*nz];
                output[(nzz - i - 1) + j*nzz + k*nxx*nzz] = input[(nz - 1) + (j - nb)*nz + (k - nb)*nx*nz];
            }
        }
    }

    for (int k = 0; k < nyy; k++)
    {   
        for (int j = 0; j < nb; j++)
        {
            for (int i = 0; i < nzz; i++)
            {
                output[i + j*nzz + k*nxx*nzz] = output[i + nb*nzz + k*nxx*nzz];
                output[i + (nxx - j - 1)*nzz + k*nxx*nzz] = output[i + (nxx - nb - 1)*nzz + k*nxx*nzz];
            }
        }
    }

    for (int k = 0; k < nb; k++)
    {   
        for (int j = 0; j < nxx; j++)
        {
            for (int i = 0; i < nzz; i++)
            {
                output[i + j*nzz + k*nxx*nzz] = output[i + j*nzz + nb*nxx*nzz];
                output[i + j*nzz + (nyy - k - 1)*nxx*nzz] = output[i + j*nzz + (nyy - nb - 1)*nxx*nzz];
            }
        }
    }
}

void Modeling::reduce_boundary(float * input, float * output)
{
    # pragma omp parallel for
    for (int index = 0; index < nPoints; index++)
    {
        int k = (int) (index / (nx*nz));         
        int j = (int) (index - k*nx*nz) / nz;    
        int i = (int) (index - j*nz - k*nx*nz);  

        output[i + j*nz + k*nx*nz] = input[(i + nb) + (j + nb)*nzz + (k + nb)*nxx*nzz];
    }
}

void Modeling::show_information()
{
    auto clear = system("clear");

    padding = (WIDTH - title.length() + 8) / 2;

    std::string line(WIDTH, '-');

    std::cout << line << '\n';
    std::cout << std::string(padding, ' ') << title << '\n';
    std::cout << line << '\n';
    
    std::cout << "Model dimensions: (z = " << (nz - 1)*dh << ", x = " << (nx - 1)*dh << ", y = " << (ny - 1)*dh << ") m\n\n";

    std::cout << "Running shot " << srcId + 1 << " of " << geometry->nrel << " in total\n\n";

    std::cout << "Current shot position: (z = " << geometry->zsrc[geometry->sInd[srcId]] << 
                                       ", x = " << geometry->xsrc[geometry->sInd[srcId]] << 
                                       ", y = " << geometry->ysrc[geometry->sInd[srcId]] << ") m\n";
}

void Modeling::initialization()
{
    float sx = geometry->xsrc[geometry->sInd[srcId]];
    float sy = geometry->ysrc[geometry->sInd[srcId]];
    float sz = geometry->zsrc[geometry->sInd[srcId]];

    sIdx = (int)(sx / dh);
    sIdy = (int)(sy / dh);
    sIdz = (int)(sz / dh);

    float * h_sw = new float[NW*NW*NW]();

    auto sw = hicks_weights(sx, sy, sz, sIdx, sIdy, sIdz, dh);

    for (int yId = 0; yId < NW; yId++)
        for (int xId = 0; xId < NW; xId++)
            for (int zId = 0; zId < NW; zId++)
                h_sw[zId + xId*NW + yId*NW*NW] = sw[zId][xId][yId];

    sIdx += nb; 
    sIdy += nb; 
    sIdz += nb;

    int * h_rIdx = new int[geometry->spread]();
    int * h_rIdy = new int[geometry->spread]();
    int * h_rIdz = new int[geometry->spread]();

    float * h_rw = new float[NW*NW*NW*geometry->spread]();

    int spreadId = 0;

    for (int recId = geometry->iRec[srcId]; recId < geometry->fRec[srcId]; recId++)
    {
        float rx = geometry->xrec[recId];
        float ry = geometry->yrec[recId];
        float rz = geometry->zrec[recId];

        int rIdx = (int)(rx / dh);
        int rIdy = (int)(ry / dh);
        int rIdz = (int)(rz / dh);

        auto rw = hicks_weights(rx, ry, rz, rIdx, rIdy, rIdz, dh);

        for (int yId = 0; yId < NW; yId++)
            for (int xId = 0; xId < NW; xId++)
                for (int zId = 0; zId < NW; zId++)
                    h_rw[zId + xId*NW + yId*NW*NW + spreadId*NW*NW*NW] = rw[zId][xId][yId];
        
        h_rIdx[spreadId] = rIdx + nb;
        h_rIdy[spreadId] = rIdy + nb;
        h_rIdz[spreadId] = rIdz + nb;

        ++spreadId;
    }

    cudaMemcpy(d_sw, h_sw, NW*NW*NW*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rw, h_rw, NW*NW*NW*geometry->spread*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_rIdx, h_rIdx, geometry->spread*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rIdy, h_rIdy, geometry->spread*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rIdz, h_rIdz, geometry->spread*sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_sw;
    delete[] h_rw;
    delete[] h_rIdx;
    delete[] h_rIdy;    
    delete[] h_rIdz;    
}

void Modeling::get_seismogram()
{
    cudaMemcpy(seismogram, d_seismogram, nt*geometry->spread*sizeof(float), cudaMemcpyDeviceToHost);
    std::string data_file = data_folder + "seismogram_nt" + std::to_string(nt) + "_nr" + std::to_string(geometry->spread) + "_" + std::to_string(int(1e6f*dt)) + "us_shot_" + std::to_string(srcId+1) + ".bin";
    export_binary_float(data_file, seismogram, nt*geometry->spread);    
}

void Modeling::forward_solver()
{
    cudaMemset(d_P, 0.0f, volsize*sizeof(float));
    cudaMemset(d_Pold, 0.0f, volsize*sizeof(float));

    for (int tId = 0; tId < tlag + nt; tId++)
    {
        compute_pressure<<<nBlocks, NTHREADS>>>(d_Vp, d_P, d_Pold, d_wavelet, d_sw, d_b1d, d_b2d, d_b3d, sIdx, sIdy, sIdz, tId, nt, nb, nxx, nyy, nzz, dh, dt, ABC);
        
        compute_seismogram<<<sBlocks, NTHREADS>>>(d_P, d_rIdx, d_rIdy, d_rIdz, d_rw, d_seismogram, geometry->spread, tId, tlag, nt, nxx, nzz);     

        std::swap(d_P, d_Pold);
    }
}

__global__ void compute_pressure(float * Vp, float * P, float * Pold, float * d_wavelet, float * d_sw, float * d_b1d, float * d_b2d, float * d_b3d, int sIdx, int sIdy, int sIdz, int tId, int nt, int nb, int nxx, int nyy, int nzz, float dh, float dt, bool ABC)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx*nzz); 

    if ((index == 0) && (tId < nt))
    {
        for (int yId = 0; yId < NW; yId++)
        {
            int yi = sIdy + yId - 3;
            for (int xId = 0; xId < NW; xId++)
            {
                int xi = sIdx + xId - 3;        
                for (int zId = 0; zId < NW; zId++)
                {
                    int zi = sIdz + zId - 3;        

                    P[zi + xi*nzz + yi*nxx*nzz] += d_sw[zId + xId*NW + yId*NW*NW]*d_wavelet[tId] / (dh*dh*dh); 
                }
            }
        }
    }

    if((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4) && (k > 3) && (k < nyy-4)) 
    {
        float d2P_dx2 = (- 9.0f*(P[i + (j-4)*nzz + k*nxx*nzz] + P[i + (j+4)*nzz + k*nxx*nzz])
                     +   128.0f*(P[i + (j-3)*nzz + k*nxx*nzz] + P[i + (j+3)*nzz + k*nxx*nzz])
                     -  1008.0f*(P[i + (j-2)*nzz + k*nxx*nzz] + P[i + (j+2)*nzz + k*nxx*nzz])
                     +  8064.0f*(P[i + (j-1)*nzz + k*nxx*nzz] + P[i + (j+1)*nzz + k*nxx*nzz])
                     - 14350.0f*(P[i + j*nzz + k*nxx*nzz]))/(5040.0f*dh*dh);

        float d2P_dy2 = (- 9.0f*(P[i + j*nzz + (k-4)*nxx*nzz] + P[i + j*nzz + (k+4)*nxx*nzz])
                     +   128.0f*(P[i + j*nzz + (k-3)*nxx*nzz] + P[i + j*nzz + (k+3)*nxx*nzz])
                     -  1008.0f*(P[i + j*nzz + (k-2)*nxx*nzz] + P[i + j*nzz + (k+2)*nxx*nzz])
                     +  8064.0f*(P[i + j*nzz + (k-1)*nxx*nzz] + P[i + j*nzz + (k+1)*nxx*nzz])
                     - 14350.0f*(P[i + j*nzz + k*nxx*nzz]))/(5040.0f*dh*dh);

        float d2P_dz2 = (- 9.0f*(P[(i-4) + j*nzz + k*nxx*nzz] + P[(i+4) + j*nzz + k*nxx*nzz])
                     +   128.0f*(P[(i-3) + j*nzz + k*nxx*nzz] + P[(i+3) + j*nzz + k*nxx*nzz])
                     -  1008.0f*(P[(i-2) + j*nzz + k*nxx*nzz] + P[(i+2) + j*nzz + k*nxx*nzz])
                     +  8064.0f*(P[(i-1) + j*nzz + k*nxx*nzz] + P[(i+1) + j*nzz + k*nxx*nzz])
                     - 14350.0f*(P[i + j*nzz + k*nxx*nzz]))/(5040.0f*dh*dh);

        Pold[index] = dt*dt*Vp[index]*Vp[index]*(d2P_dx2 + d2P_dy2 + d2P_dz2) + 2.0f*P[index] - Pold[index];
        
        if (ABC)
        {
            float damper = get_boundary_damper(d_b1d, d_b2d, d_b3d, i, j, k, nxx, nyy, nzz, nb);

            P[index] *= damper;
            Pold[index] *= damper;
        }
    }
}

__global__ void compute_seismogram(float * P, int * d_rIdx, int * d_rIdy, int * d_rIdz, float * d_rw, float * seismogram, int spread, int tId, int tlag, int nt, int nxx, int nzz)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if ((index < spread) && (tId > tlag))
    {   
        seismogram[(tId - tlag) + index*nt] = 0.0f;    
                
        for (int yId = 0; yId < NW; yId++)
        {
            int yi = d_rIdy[index] + yId - 3;
            for (int xId = 0; xId < NW; xId++)
            {
                int xi = d_rIdx[index] + xId - 3;        
                for (int zId = 0; zId < NW; zId++)
                {
                    int zi = d_rIdz[index] + zId - 3;        

                    seismogram[(tId - tlag) + index*nt] += d_rw[zId + xId*NW + yId*NW*NW + index*NW*NW*NW]*P[zi + xi*nzz + yi*nxx*nzz];
                }
            }
        }
    } 
}

__device__ float get_boundary_damper(float * damp1D, float * damp2D, float * damp3D, int i, int j, int k, int nxx, int nyy, int nzz, int nb)
{
    float damper;

    // global case
    if ((i >= nb) && (i < nzz-nb) && (j >= nb) && (j < nxx-nb) && (k >= nb) && (k < nyy-nb))
    {
        damper = 1.0f;
    }

    // 1D damping
    else if((i < nb) && (j >= nb) && (j < nxx-nb) && (k >= nb) && (k < nyy-nb)) 
    {
        damper = damp1D[i];
    }         
    else if((i >= nzz-nb) && (i < nzz) && (j >= nb) && (j < nxx-nb) && (k >= nb) && (k < nyy-nb)) 
    {
        damper = damp1D[nb-(i-(nzz-nb))-1];
    }         
    else if((i >= nb) && (i < nzz-nb) && (j >= 0) && (j < nb) && (k >= nb) && (k < nyy-nb)) 
    {
        damper = damp1D[j];
    }
    else if((i >= nb) && (i < nzz-nb) && (j >= nxx-nb) && (j < nxx) && (k >= nb) && (k < nyy-nb)) 
    {
        damper = damp1D[nb-(j-(nxx-nb))-1];
    }
    else if((i >= nb) && (i < nzz-nb) && (j >= nb) && (j < nxx-nb) && (k >= 0) && (k < nb)) 
    {
        damper = damp1D[k];
    }
    else if((i >= nb) && (i < nzz-nb) && (j >= nb) && (j < nxx-nb) && (k >= nyy-nb) && (k < nyy)) 
    {
        damper = damp1D[nb-(k-(nyy-nb))-1];
    }

    // 2D damping 
    else if((i >= nb) && (i < nzz-nb) && (j >= 0) && (j < nb) && (k >= 0) && (k < nb))
    {
        damper = damp2D[j + k*nb];
    }
    else if((i >= nb) && (i < nzz-nb) && (j >= nxx-nb) && (j < nxx) && (k >= 0) && (k < nb))
    {
        damper = damp2D[nb-(j-(nxx-nb))-1 + k*nb];
    }
    else if((i >= nb) && (i < nzz-nb) && (j >= 0) && (j < nb) && (k >= nyy-nb) && (k < nyy))
    {
        damper = damp2D[j + (nb-(k-(nyy-nb))-1)*nb];
    }
    else if((i >= nb) && (i < nzz-nb) && (j >= nxx-nb) && (j < nxx) && (k >= nyy-nb) && (k < nyy))
    {
        damper = damp2D[nb-(j-(nxx-nb))-1 + (nb-(k-(nyy-nb))-1)*nb];
    }

    else if((i >= 0) && (i < nb) && (j >= nb) && (j < nxx-nb) && (k >= 0) && (k < nb))
    {
        damper = damp2D[i + k*nb];
    }
    else if((i >= nzz-nb) && (i < nzz) && (j >= nb) && (j < nxx-nb) && (k >= 0) && (k < nb))
    {
        damper = damp2D[nb-(i-(nzz-nb))-1 + k*nb];
    }
    else if((i >= 0) && (i < nb) && (j >= nb) && (j < nxx-nb) && (k >= nyy-nb) && (k < nyy))
    {
        damper = damp2D[i + (nb-(k-(nyy-nb))-1)*nb];
    }
    else if((i >= nzz-nb) && (i < nzz) && (j >= nb) && (j < nxx-nb) && (k >= nyy-nb) && (k < nyy))
    {
        damper = damp2D[nb-(i-(nzz-nb))-1 + (nb-(k-(nyy-nb))-1)*nb];
    }

    else if((i >= 0) && (i < nb) && (j >= 0) && (j < nb) && (k >= nb) && (k < nyy-nb))
    {
        damper = damp2D[i + j*nb];
    }
    else if((i >= nzz-nb) && (i < nzz) && (j >= 0) && (j < nb) && (k >= nb) && (k < nyy-nb))
    {
        damper = damp2D[nb-(i-(nzz-nb))-1 + j*nb];
    }
    else if((i >= 0) && (i < nb) && (j >= nxx-nb) && (j < nxx) && (k >= nb) && (k < nyy-nb))
    {
        damper = damp2D[i + (nb-(j-(nxx-nb))-1)*nb];
    }
    else if((i >= nzz-nb) && (i < nzz) && (j >= nxx-nb) && (j < nxx) && (k >= nb) && (k < nyy-nb))
    {
        damper = damp2D[nb-(i-(nzz-nb))-1 + (nb-(j-(nxx-nb))-1)*nb];
    }

    // 3D damping
    else if((i >= 0) && (i < nb) && (j >= 0) && (j < nb) && (k >= 0) && (k < nb))
    {
        damper = damp3D[i + j*nb + k*nb*nb];
    }
    else if((i >= nzz-nb) && (i < nzz) && (j >= 0) && (j < nb) && (k >= 0) && (k < nb))
    {
        damper = damp3D[nb-(i-(nzz-nb))-1 + j*nb + k*nb*nb];
    }
    else if((i >= 0) && (i < nb) && (j >= nxx-nb) && (j < nxx) && (k >= 0) && (k < nb))
    {
        damper = damp3D[i + (nb-(j-(nxx-nb))-1)*nb + k*nb*nb];
    }
    else if((i >= 0) && (i < nb) && (j >= 0) && (j < nb) && (k >= nyy-nb) && (k < nyy))
    {
        damper = damp3D[i + j*nb + (nb-(k-(nyy-nb))-1)*nb*nb];
    }
    else if((i >= nzz-nb) && (i < nzz) && (j >= nxx-nb) && (j < nxx) && (k >= 0) && (k < nb))
    {
        damper = damp3D[nb-(i-(nzz-nb))-1 + (nb-(j-(nxx-nb))-1)*nb + k*nb*nb];
    }
    else if((i >= nzz-nb) && (i < nzz) && (j >= 0) && (j < nb) && (k >= nyy-nb) && (k < nyy))
    {
        damper = damp3D[nb-(i-(nzz-nb))-1 + j*nb + (nb-(k-(nyy-nb))-1)*nb*nb];
    }
    else if((i >= 0) && (i < nb) && (j >= nxx-nb) && (j < nxx) && (k >= nyy-nb) && (k < nyy))
    {
        damper = damp3D[i + (nb-(j-(nxx-nb))-1)*nb + (nb-(k-(nyy-nb))-1)*nb*nb];
    }
    else if((i >= nzz-nb) && (i < nzz) && (j >= nxx-nb) && (j < nxx) && (k >= nyy-nb) && (k < nyy))
    {
        damper = damp3D[nb-(i-(nzz-nb))-1 + (nb-(j-(nxx-nb))-1)*nb + (nb-(k-(nyy-nb))-1)*nb*nb];
    }

    return damper;
}

// Random Boundary Condition

std::random_device RBC;  
std::mt19937 RBC_RNG(RBC()); 

void Modeling::set_coordinates()
{
    float * h_Z = new float[nzz]();   
    # pragma omp parallel for 
    for (int i = 0; i < nzz; i++) 
        h_Z[i] = (float)(i)*dh;

    float * h_X = new float[nxx]();   
    # pragma omp parallel for 
    for (int j = 0; j < nxx; j++) 
        h_X[j] = (float)(j)*dh;

    float * h_Y = new float[nyy]();   
    # pragma omp parallel for 
    for (int k = 0; k < nyy; k++) 
        h_Y[k] = (float)(k)*dh;
        
    cudaMalloc((void**)&(d_X), nxx*sizeof(float));
    cudaMemcpy(d_X, h_X, nxx*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(d_Y), nyy*sizeof(float));
    cudaMemcpy(d_Y, h_Y, nyy*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(d_Z), nzz*sizeof(float));
    cudaMemcpy(d_Z, h_Z, nzz*sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_X;
    delete[] h_Y;
    delete[] h_Z;
}

void Modeling::set_random_boundary(float * vp, float ratio, float varVp)
{
    float x_max = (nxx-1)*dh;
    float y_max = (nyy-1)*dh;
    float z_max = (nzz-1)*dh;

    random_boundary_bg<<<nBlocks,NTHREADS>>>(vp, nxx, nyy, nzz, nb, varVp);

    std::vector<Point> points = poissonDiskSampling(x_max, y_max, z_max, ratio);
    std::vector<Point> target;
    
    for (int index = 0; index < points.size(); index++)
    {
        if (!((points[index].x > 0.5f*nb*dh) && (points[index].x < x_max - 0.5f*nb*dh) &&
              (points[index].y > 0.5f*nb*dh) && (points[index].y < y_max - 0.5f*nb*dh) &&
              (points[index].z > 0.5f*nb*dh) && (points[index].z < z_max - 0.5f*nb*dh)))
            target.push_back(points[index]);
    }
    
    for (int p = 0; p < target.size(); p++)
    {
        float xc = target[p].x;
        float yc = target[p].y;
        float zc = target[p].z;

        float r = std::uniform_real_distribution<float>(0.5f*ratio, ratio)(RBC_RNG);
        float A = std::uniform_real_distribution<float>(0.5f*varVp, varVp)(RBC_RNG);

        float factor = rand() % 2 == 0 ? -1.0f : 1.0f;

        random_boundary_gp<<<nBlocks,NTHREADS>>>(vp, d_X, d_Y, d_Z, nxx, nyy, nzz, x_max, y_max, z_max, nb, dh, factor, A, xc, yc, zc, r, vmax, vmin, varVp);
    }
}

__global__ void random_boundary_gp(float * vp, float * X, float * Y, float * Z, int nxx, int nyy, int nzz, float x_max, float y_max, float z_max, int nb, float dh, float factor, float A, float xc, float yc, float zc, float r, float vmax, float vmin, float varVp)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx*nzz);    
    
    if (!((X[j] > (float)(nb)*dh) && (X[j] < x_max - (float)(nb)*dh) &&
          (Y[k] > (float)(nb)*dh) && (Y[k] < y_max - (float)(nb)*dh) &&
          (Z[i] > (float)(nb)*dh) && (Z[i] < z_max - (float)(nb)*dh)))
    {
        vp[index] += factor*A*expf(-0.5f*(((X[j]-xc)/r)*((X[j]-xc)/r) +
                                          ((Y[k]-yc)/r)*((Y[k]-yc)/r) +  
                                          ((Z[i]-zc)/r)*((Z[i]-zc)/r)));
        
        vp[index] = vp[index] > vmax + varVp ? vmax + varVp : vp[index];
        vp[index] = vp[index] < vmin - varVp ? vmin - varVp : vp[index];         
    }   
}

__global__ void random_boundary_bg(float * vp, int nxx, int nyy, int nzz, int nb, float varVp)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int k = (int) (index / (nxx*nzz));         
    int j = (int) (index - k*nxx*nzz) / nzz;   
    int i = (int) (index - j*nzz - k*nxx*nzz); 

    if (!((i >= nb) && (i < nzz - nb) &&
          (j >= nb) && (j < nxx - nb) &&
          (k >= nb) && (k < nyy - nb)))
    {
        int ii = min(max(i, nb), nzz - nb - 1);
        int jj = min(max(j, nb), nxx - nb - 1);
        int kk = min(max(k, nb), nyy - nb - 1);

        int index_ref = ii + jj*nzz + kk*nxx*nzz;

        float val = vp[index_ref];

        if (j < nb) {
            float fx = 1.0f - (float)(j) / (float)(nb);
            val = get_random_value(val, fx, varVp, index);
        } 
        else if (j >= nxx - nb) {
            float fx = 1.0f - (float)(nxx - j - 1) / (float)(nb);
            val = get_random_value(val, fx, varVp, index);
        }

        if (k < nb) {
            float fy = 1.0f - (float)(k) / (float)(nb);
            val = get_random_value(val, fy, varVp, index);
        } 
        else if (k >= nyy - nb) {
            float fy = 1.0f - (float)(nyy - k - 1) / (float)(nb);
            val = get_random_value(val, fy, varVp, index);
        }

        if (i < nb) {
            float fz = 1.0f - (float)(i) / (float)(nb);
            val = get_random_value(val, fz, varVp, index);
        } 
        else if (i >= nzz - nb) {
            float fz = 1.0f - (float)(nzz - i - 1) / (float)(nb);
            val = get_random_value(val, fz, varVp, index);
        }

        vp[index] = val;        
    }
}

__device__ float get_random_value(float velocity, float function, float parameter, int index)
{
    curandState state;
    curand_init(clock64(), index, 0, &state);
    return velocity + function*parameter*curand_normal(&state);
}