# ifndef INVERSION_CUH
# define INVERSION_CUH

# include "../modeling/modeling.cuh"

class Inversion : public Modeling
{
private:

    int iteration;
    int max_iteration;

    float alpha;
    float beta1;
    float beta2;

    float rbc_ratio;
    float rbc_varVp;
    float rbc_length; 

    int abc_nxx, abc_nzz, abc_nb, abc_matsize;
    int rbc_nxx, rbc_nzz, rbc_nb, rbc_matsize;

    float * vp = nullptr;
    float * A1 = nullptr;
    float * A2 = nullptr;

    float * sumPs = nullptr;
    float * partial = nullptr;    
    float * gradient = nullptr;

    float * h_rbc_Vp = nullptr;
    float * d_rbc_Vp = nullptr;

    float * obs_data = nullptr;

    float * d_Ps = nullptr;    
    float * d_Pr = nullptr;
    float * d_Psold = nullptr;
    float * d_Prold = nullptr;
    float * d_sumPs = nullptr;
    float * d_gradient = nullptr;

    std::string stage_info;
    
    std::string model_file;
    std::string input_folder;
    std::string output_folder; 
    std::string residuo_folder; 
    
    std::vector<float> residuo;
    
    void rbc_forward_solver();
    void set_seismic_source();
    void forward_propagation();
    void backward_propagation();

public:

    int freqId;
    int nFreqs;

    bool converged;

    void set_parameters();
    void set_observed_data();
    void show_information();
    void check_convergence();
    void set_calculated_data();
    void compute_gradient();
    void optimization();
    void update_model();

    void export_final_model();
    void export_convergence();
};

__global__ void FWI(float * Ps, float * Psold, float * Pr, float * Prold, float * Vp, float * seismogram, float * gradient, float * sumPs, int * rIdx, int * rIdz, int spread, int tId, int tlag, int nxx, int nzz, int nt, float dh, float dt);

# endif