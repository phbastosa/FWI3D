# include "migration/migration.cuh"

int main(int argc, char **argv)
{
    auto migration = new Migration();

    migration->parameters = std::string(argv[1]);

    migration->set_parameters();

    auto ti = std::chrono::system_clock::now();

    for (int srcId = 0; srcId < migration->geometry->nrel; srcId++)
    {
        migration->srcId = srcId;

        migration->forward_propagation();
        migration->backward_propagation();        
    }

    auto tf = std::chrono::system_clock::now();

    migration->export_seismic();

    std::chrono::duration<double> elapsed_seconds = tf - ti;
    std::cout << "\nRun time: " << elapsed_seconds.count() << " s." << std::endl;
    
    return 0;
}