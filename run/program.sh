#!/bin/bash

admin="../src/admin/admin.cpp"

geometry="../src/geometry/geometry.cpp"

modeling="../src/modeling/modeling.cu"
modeling_main="../src/modeling_main.cpp"

inversion="../src/inversion/inversion.cu"
inversion_main="../src/inversion_main.cpp"

migration="../src/migration/migration.cu"
migration_main="../src/migration_main.cpp"

flags="-Xcompiler -fopenmp --std=c++11 -lm -lfftw3 -O3"

# Main dialogue ---------------------------------------------------------------------------------------

USER_MESSAGE="
-------------------------------------------------------------------------------
                                    \033[34mFWI\033[0;0m
-------------------------------------------------------------------------------
\nUsage:\n
    $ $0 -modeling                      
    $ $0 -inversion           
    $ $0 -migration
\nTests:\n
    $ $0 -test_modeling                      
    $ $0 -test_inversion           
    $ $0 -test_migration
    
-------------------------------------------------------------------------------
"

[ -z "$1" ] && 
{
	echo -e "\nYou didn't provide any parameter!" 
	echo -e "Type $0 -help for more info\n"
    exit 1 
}

case "$1" in

-h) 

	echo -e "$USER_MESSAGE"
	exit 0
;;

-compile) 

    echo -e "Compiling stand-alone executables!\n"

    echo -e "../bin/\033[31mmodeling.exe\033[m" 
    nvcc $admin $geometry $modeling $modeling_main $flags -o ../bin/modeling.exe

    # echo -e "../bin/\033[31minversion.exe\033[m" 
    # nvcc $admin $geometry $modeling $inversion $inversion_main $flags -o ../bin/inversion.exe

    # echo -e "../bin/\033[31mmigration.exe\033[m"
    # nvcc $admin $geometry $modeling $migration $migration_main $flags -o ../bin/migration.exe

    exit 0
;;

-modeling) 

    ./../bin/modeling.exe parameters.txt
	
    exit 0
;;

-inversion) 
    
    ./../bin/inversion.exe parameters.txt
	
    exit 0
;;

-migration) 
    
    ./../bin/migration.exe parameters.txt
	
    exit 0
;;

-test_modeling)

    prefix=../tests/modeling
    parameters=$prefix/parameters.txt

    python3 -B $prefix/generate_models.py
    python3 -B $prefix/generate_geometry.py

    ./../bin/modeling.exe $parameters

    python3 -B $prefix/generate_figures.py $parameters

	exit 0
;;

-test_inversion) 

    # prefix=../tests/inversion
    # parameters=$prefix/parameters.txt

    # python3 -B $prefix/generate_models.py
    # python3 -B $prefix/generate_geometry.py
 
    # true_model="model_file = ../inputs/models/inversion_test_true_model_201x501_10m.bin"
    # init_model="model_file = ../inputs/models/inversion_test_init_model_201x501_10m.bin"

    # ./../bin/modeling.exe $parameters

    # sed -i "s|$true_model|$init_model|g" "$parameters"

    # ./../bin/inversion.exe $parameters

    # sed -i "s|$init_model|$true_model|g" "$parameters"

    # python3 -B $prefix/generate_figures.py $parameters

    exit 0
;;

-test_migration)

    prefix=../tests/migration
    parameters=$prefix/parameters.txt

    python3 -B $prefix/generate_models.py
    python3 -B $prefix/generate_geometry.py

    ./../bin/modeling.exe $parameters

    python3 -B $prefix/generate_input_data.py $parameters

    ./../bin/migration.exe $parameters

    python3 -B $prefix/generate_figures.py $parameters

	exit 0
;;

-clean)

    rm *.png

    rm ../bin/*.exe
    rm ../inputs/data/*.bin
    rm ../inputs/models/*.bin
    rm ../inputs/geometry/*.txt

    rm ../outputs/data/*.bin
    rm ../outputs/models/*.bin
    rm ../outputs/seismic/*.bin
    rm ../outputs/residuo/*.txt

;;

* ) 

	echo -e "\033[31mERRO: Option $1 unknown!\033[m"
	echo -e "\033[31mType $0 -h for help \033[m"
	
    exit 3
;;

esac