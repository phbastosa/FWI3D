# ifndef MIGRATION_CUH
# define MIGRATION_CUH

# include "../modeling/modeling.cuh"

class Migration : public Modeling
{
private:

    float rbc_ratio;
    float rbc_varVp;
    float rbc_length; 

    float * d_Pr = nullptr;
    float * d_Prold = nullptr;
    float * d_image = nullptr;
    float * d_sumPs = nullptr;

    float * image = nullptr;
    float * sumPs = nullptr;
    float * partial = nullptr;

    std::string stage_info;
    std::string input_folder;
    std::string output_folder;
    
    void show_information();
    void set_seismic_source();

public:

    void set_parameters();
    void forward_propagation();
    void backward_propagation();
    void export_seismic();
};

__global__ void RTM(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * seismogram, float * image, float * sumPs, int * rIdx, int * rIdz, int spread, int tId, int tlag, int nxx, int nzz, int nt, float dx, float dt);

# endif