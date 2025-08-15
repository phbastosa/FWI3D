# include "modeling/modeling.cuh"

int main(int argc, char **argv)
{
    auto modeling = new Modeling();

    modeling->parameters = std::string(argv[1]);

    modeling->set_parameters();

    auto ti = std::chrono::system_clock::now();

    // for (int shot = 0; shot < modeling->geometry->nrel; shot++)
    // {
    //     modeling->srcId = shot;

    //     modeling->show_information();

    //     modeling->initialization();
    //     modeling->forward_solver();
    //     modeling->set_seismogram();
    // }

    auto tf = std::chrono::system_clock::now();

    // modeling->export_output_data();

    std::chrono::duration<double> elapsed_seconds = tf - ti;
    std::cout << "\nRun time: " << elapsed_seconds.count() << " s." << std::endl;
    
    return 0;
}