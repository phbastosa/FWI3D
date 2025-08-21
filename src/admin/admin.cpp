# include "admin.hpp"

bool str2bool(std::string s)
{
    bool b;

    std::for_each(s.begin(), s.end(), [](char & c){c = ::tolower(c);});
    std::istringstream(s) >> std::boolalpha >> b;

    return b;
}

void import_binary_float(std::string path, float * array, int n)
{
    std::ifstream file(path, std::ios::in);

    if (!file.is_open())
        throw std::invalid_argument("Error: \033[31m" + path + "\033[0;0m could not be opened!");
    
    file.read((char *) array, n * sizeof(float));
    
    file.close();    
}

void export_binary_float(std::string path, float * array, int n)
{
    std::ofstream file(path, std::ios::out);
    
    if (!file.is_open()) 
        throw std::invalid_argument("Error: \033[31m" + path + "\033[0;0m could not be opened!");

    file.write((char *) array, n * sizeof(float));

    std::cout<<"\nBinary file \033[34m" + path + "\033[0;0m was successfully written."<<std::endl;

    file.close();
}

void import_text_file(std::string path, std::vector<std::string> &elements)
{
    std::ifstream file(path, std::ios::in);
    
    if (!file.is_open()) 
        throw std::invalid_argument("Error: \033[31m" + path + "\033[0;0m could not be opened!");

    std::string line;
    while(getline(file, line))
        if (line[0] != '#') elements.push_back(line);

    file.close();
}

std::string catch_parameter(std::string target, std::string file)
{
    char spaces = ' ';
    char comment = '#';

    std::string line;
    std::string variable;

    std::ifstream parameters(file);

    if (!parameters.is_open()) 
        throw std::invalid_argument("Error: \033[31m" + file + "\033[0;0m could not be opened!");

    while (getline(parameters, line))
    {           
        if ((line.front() != comment) && (line.front() != spaces))        
        {
            if (line.find(target) == 0)
            {
                for (int i = line.find("=")+2; i < line.size(); i++)
                {    
                    if (line[i] == '#') break;
                    variable += line[i];            
                }

                break;
            }
        }                 
    }
    
    parameters.close();

    variable.erase(remove(variable.begin(), variable.end(), ' '), variable.end());

    return variable;
}

std::vector<std::string> split(std::string s, char delimiter)
{
    std::string token;
    std::vector<std::string> tokens;
    std::istringstream tokenStream(s);

    while (getline(tokenStream, token, delimiter)) 
        tokens.push_back(token);
   
    return tokens;
}

float sinc(float x) 
{
    if (fabsf(x) < 1e-8f) return 1.0f;
    return sinf(M_PI * x) / (M_PI * x);
}

float bessel_i0(float x) 
{
    float sum = 1.0f, term = 1.0f, k = 1.0f;
    
    while (term > 1e-10f) 
    {
        term *= (x / (2.0f * k)) * (x / (2.0f * k));
        sum += term;
        k += 1.0f;
    }

    return sum;
}

std::vector<std::vector<std::vector<float>>> hicks_weights(float x, float y, float z, int ix0, int iy0, int iz0, float dh) 
{
    const int N = 8;
    const float beta = 6.41f;

    std::vector<std::vector<std::vector<float>>> weights(N, std::vector<std::vector<float>>(N, std::vector<float>(N)));

    float rmax = sqrtf(2.0f*dh*dh);
    float I0_beta = bessel_i0(beta);

    float sum = 0.0f;

    for (int k = 0; k < N; ++k) 
    {
        float yk = (iy0 + k - 3) * dh;
        float dyr = (y - yk) / dh;

        for (int j = 0; j < N; ++j) 
        {
            float xj = (ix0 + j - 3) * dh;
            float dxr = (x - xj) / dh;

            for (int i = 0; i < N; ++i) 
            {    
                float zi = (iz0 + i - 3) * dh;
                float dzr = (z - zi) / dh;
        
                float rz = z - zi;
                float rx = x - xj;
                float ry = y - yk;

                float r = sqrtf(rx*rx + ry*ry + rz*rz);

                float rnorm = r / rmax;

                float wij = 0.0f;
                if (rnorm <= 1.0f) 
                {
                    float arg = beta * sqrtf(1.0f - rnorm * rnorm);
                    wij = bessel_i0(arg) / I0_beta;
                }

                float sinc_term = sinc(dxr) * sinc(dyr) * sinc(dzr);

                weights[i][j][k] = sinc_term * wij;

                sum += weights[i][j][k];
            }
        }
    }

    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                weights[i][j][k] /= sum;

    return weights;
}

std::random_device rd;  
std::mt19937 rng(rd()); 

std::vector<Point> poissonDiskSampling(float x_max, float y_max, float z_max, float radius)
{
    float cell_size = radius / sqrtf(3.0f);

    int grid_x = static_cast<int>(ceilf(x_max / cell_size));
    int grid_y = static_cast<int>(ceilf(y_max / cell_size));
    int grid_z = static_cast<int>(ceilf(z_max / cell_size));

    std::vector<Point> points;
    std::vector<Point> activeList;

    std::vector<Point> grid(grid_x*grid_y*grid_z, {-1, -1, -1});

    std::uniform_real_distribution<float> distX(0.0f, x_max);
    std::uniform_real_distribution<float> distY(0.0f, y_max);
    std::uniform_real_distribution<float> distZ(0.0f, z_max);

    auto insertIntoGrid = [&](const Point& p) 
    {
        int gx = static_cast<int>(p.x / cell_size);
        int gy = static_cast<int>(p.y / cell_size);
        int gz = static_cast<int>(p.z / cell_size);
        
        grid[gz + gx*grid_z + gy*grid_x*grid_z] = p;
    };

    auto isValid = [&](const Point& p) 
    {
        if (p.x < 0.0f || p.y < 0 || p.z < 0 || p.x >= x_max || p.y >= y_max || p.z >= z_max) return false;
    
        int gx = static_cast<int>(p.x / cell_size);
        int gy = static_cast<int>(p.y / cell_size);
        int gz = static_cast<int>(p.z / cell_size);

        for (int k = -1; k <= 1; k++) 
        {
            for (int j = -1; j <= 1; j++) 
            {
                for (int i = -1; i <= 1; i++) 
                {
                    int nx = gx + j;
                    int ny = gy + k;
                    int nz = gz + i;
                
                    if (nx >= 0 && ny >= 0 && nz >= 0 && nx < grid_x && ny < grid_y && nz < grid_z) 
                    {
                        Point neighbor = grid[nz + nx*grid_z + ny*grid_x*grid_z];
                        
                        float dx = neighbor.x - p.x;    
                        float dy = neighbor.y - p.y;    
                        float dz = neighbor.z - p.z;    

                        if (neighbor.x != -1 && sqrtf(dx*dx + dy*dy + dz*dz) < radius) 
                            return false;
                    }
                }
            }
        }
        return true;
    };

    auto generateRandomPointAround = [&](const Point& p) 
    {
        std::uniform_real_distribution<float> distTheta(0, 2 * M_PI);
        std::uniform_real_distribution<float> distPhi(0, M_PI);
        std::uniform_real_distribution<float> distRadius(radius, 2 * radius);
        
        float theta = distTheta(rng);
        float phi = distPhi(rng);
        float r = distRadius(rng);
        
        return Point{
            p.x + r * sinf(phi) * cosf(theta),
            p.y + r * sinf(phi) * sinf(theta),
            p.z + r * cosf(phi)
        };
    };

    Point initial = {distX(rng), distY(rng), distZ(rng)};
    points.push_back(initial);
    activeList.push_back(initial);
    insertIntoGrid(initial);
    
    while (!activeList.empty()) 
    {
        int index = std::uniform_int_distribution<int>(0, activeList.size() - 1)(rng);
        Point point = activeList[index];
        bool found = false;

        for (int i = 0; i < 30; ++i) 
        {
            Point newPoint = generateRandomPointAround(point);
            if (isValid(newPoint)) 
            {
                points.push_back(newPoint);
                activeList.push_back(newPoint);
                insertIntoGrid(newPoint);
                found = true;
            }
        }
        if (!found) activeList.erase(activeList.begin() + index);
    }
    return points;
}