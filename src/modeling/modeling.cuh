# ifndef MODELING_CUH
# define MODELING_CUH

# include <cuda_runtime.h>
# include <curand_kernel.h>

# include "../geometry/geometry.hpp"

# define WIDTH 80
# define NTHREADS 256

class Modeling
{
protected:

    bool ABC;

    float abc_length;
    float abc_factor;
    float vmax, vmin;
    float dh, dt, fmax;

    std::string title;
    
    int padding;

    int sBlocks, nBlocks;

    int nxx, nyy, nzz, matsize;
    int nt, nx, ny, nz, nb, nPoints;
    int sIdx, sIdy, sIdz, tlag;

    int * rIdx = nullptr;
    int * rIdy = nullptr;
    int * rIdz = nullptr;

    float * Vp = nullptr;

    float * seismogram = nullptr;

    float * d_X = nullptr;
    float * d_Y = nullptr;
    float * d_Z = nullptr;

    int * d_rIdx = nullptr;
    int * d_rIdy = nullptr;
    int * d_rIdz = nullptr;

    float * d_P = nullptr;
    float * d_Vp = nullptr;
    float * d_Pold = nullptr;

    float * d_b1d = nullptr;
    float * d_b2d = nullptr;
    float * d_b3d = nullptr;
    
    float * d_wavelet = nullptr;
    float * d_seismogram = nullptr;

    std::string data_folder;

    void set_wavelet();
    void set_geometry();
    void set_properties();
    void set_coordinates();
    void set_seismograms();    
    void set_abc_dampers();
    void set_main_parameters();

    void expand_boundary(float * input, float * output);
    void reduce_boundary(float * input, float * output);

    void set_random_boundary(float * vp, float ratio, float varVp);

public:

    int srcId;

    Geometry * geometry;

    std::string parameters;

    void set_parameters();
    void initialization();
    void forward_solver();
    void get_seismogram();
    void show_information();    
    void export_output_data();
};

__global__ void compute_pressure(float * Vp, float * P, float * Pold, float * d_wavelet, float * d_b1d, float * d_b2d, float * d_b3d, int sIdx, int sIdy, int sIdz, int tId, int nt, int nb, int nxx, int nyy, int nzz, float dh, float dt, bool ABC);
__global__ void compute_seismogram(float * P, int * d_rIdx, int * d_rIdy, int * d_rIdz, float * seismogram, int spread, int tId, int tlag, int nt, int nxx, int nzz);
__device__ float get_boundary_damper(float * damp1D, float * damp2D, float * damp3D, int i, int j, int k, int nxx, int nyy, int nzz, int nb);

// __device__ float get_random_value(float velocity, float function, float parameter, int index, float varVp);
// __global__ void random_boundary_bg(float * Vp, int nxx, int nzz, int nb, float varVp);
// __global__ void random_boundary_gp(float * Vp, float * X, float * Z, int nxx, int nzz, float x_max, float z_max, float xb, float zb, float factor, float A, float xc, float zc, float r, float vmax, float vmin, float varVp);

# endif