# include "inversion/inversion.cuh"

int main(int argc, char **argv)
{
    auto inversion = new Inversion();

    inversion->parameters = std::string(argv[1]);

    inversion->set_parameters();
    inversion->set_observed_data();

    // auto ti = std::chrono::system_clock::now();

    // while (true)
    // {
    //     inversion->set_calculated_data();

    //     inversion->check_convergence();

    //     if (inversion->converged) break;

    //     inversion->compute_gradient();
           
    //     inversion->optimization();

    //     inversion->update_model();
    // }
        
    // inversion->export_convergence();
    // inversion->export_final_model();

    // auto tf = std::chrono::system_clock::now();

    // std::chrono::duration<double> elapsed_seconds = tf - ti;
    // std::cout << "\nRun time: " << elapsed_seconds.count() << " s." << std::endl;
    
    return 0;
}