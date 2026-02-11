#include<iostream>
#include<chrono>
#include<cmath>
#include<vector>
#include<thread>
#include<fstream>
using namespace std;
using namespace std::chrono;

void compute_sequential(double *array, int array_size){
    for (int i=0; i < array_size; i++){
        array[i] = std::sin(array[i])*0.5+0.25;
    }
}
void compute_openmp(double *array, int array_size, int num_thread){
    #pragma omp parallel for num_threads(num_thread)
    for (int i=0; i < array_size; i++){
        array[i] = std::sin(array[i])*0.5+0.25;
    } 
}
void compute_thread(double *array, int start,int end){
    for (int i=start; i < end; i++){
        array[i] = std::sin(array[i])*0.5+0.25;
    }
}
void compute_threads(double *array, int array_size, int num_thread){
    vector<thread> threads;
    int chunk_size = array_size/num_thread;
    for (int i=0; i< num_thread; i++){
        int start = i*chunk_size;
        int end = (i == num_thread-1)? array_size : (i+1)*chunk_size;
        threads.push_back(thread(compute_thread, array, start, end));
    }
    for (auto &t : threads){
        t.join();
    }
}

// workload - impl - threads - size - run_i - elapsed_ms
void compute_sequential_function(int sum_run, int max_threads, int array_size, double *array, bool is_battery){
    string file_path = is_battery ? "./csv/result_sequential_sin_battery.csv" : "./csv/result_sequential_sin.csv";
    ofstream file(file_path);

    for(int run_i=1; run_i <= sum_run; run_i++){
        // Массивыг дахин эхлүүлэх
        for (int j=0; j<array_size; j++){
            array[j]=1.0;
        }
        
        auto t1 = steady_clock::now();
        compute_sequential(array, array_size);
        auto t2 = steady_clock::now();
        auto duration = duration_cast<milliseconds>(t2-t1);
        file << "sin" << "," << "sequential" << "," << 1 << "," << array_size << "," << run_i << "," << duration.count() << endl;
    }
    file.close();
}

void compute_openmp_function(int sum_run, int max_threads, int array_size, double *array, bool is_battery){
    string file_path = is_battery ? "./csv/result_openmp_sin_battery.csv" : "./csv/result_openmp_sin.csv";
    ofstream file(file_path);

    for(int run_i=1; run_i <= sum_run; run_i++){
        // Sequential (1 thread)
        for (int j=0; j<array_size; j++){
            array[j]=1.0;
        }
        auto t1 = steady_clock::now();
        compute_openmp(array, array_size, 1);
        auto t2 = steady_clock::now();
        auto duration = duration_cast<milliseconds>(t2-t1);
        file << "sin" << "," << "openmp" << "," << 1 << "," << array_size << "," << run_i << "," << duration.count() << endl;

        // Parallel (2, 4, 6, ... threads)
        for (int thread = 2; thread <= max_threads; thread += 2){
            for (int j=0; j<array_size; j++){
                array[j]=1.0;
            }
            auto t3 = steady_clock::now();
            compute_openmp(array, array_size, thread);
            auto t4 = steady_clock::now();
            auto duration_parallel = duration_cast<milliseconds>(t4-t3);
            file << "sin" << "," << "openmp" << "," << thread << "," << array_size << "," << run_i << "," << duration_parallel.count() << endl;
        }
    }
    file.close();
}

void compute_threads_function(int sum_run, int max_threads, int array_size, double *array, bool is_battery){
    string file_path = is_battery ? "./csv/result_threads_sin_battery.csv" : "./csv/result_threads_sin.csv";
    ofstream file(file_path);

    for(int run_i=1; run_i <= sum_run; run_i++){
        for (int j=0; j<array_size; j++){
            array[j]=1.0;
        }
        auto t1 = steady_clock::now();
        compute_threads(array, array_size, 1);
        auto t2 = steady_clock::now();
        auto duration = duration_cast<milliseconds>(t2-t1);
        file << "sin" << "," << "threads" << "," << 1 << "," << array_size << "," << run_i << "," << duration.count() << endl;

        for (int thread = 2; thread <= max_threads; thread += 2){
            for (int j=0; j<array_size; j++){
                array[j]=1.0;
            }
            auto t3 = steady_clock::now();
            compute_threads(array, array_size, thread);
            auto t4 = steady_clock::now();
            auto duration_parallel = duration_cast<milliseconds>(t4-t3);
            file << "sin" << "," << "threads" << "," << thread << "," << array_size << "," << run_i << "," << duration_parallel.count() << endl;
        }
    }
    file.close();
}

int main(){
    int array_size = 80000000;
    double *array = new double[array_size];
    int num_thread;
    cout << "Write num thread: ";
    cin >> num_thread;
    cout << "Num thread: " << num_thread << endl;
    
    char battery_input;
    bool is_battery = false;
    cout << "Цэнэглэж байгаа юу? (y/n): ";
    cin >> battery_input;
    if(battery_input == 'n' || battery_input == 'N'){
        is_battery = true;
        cout << "Battery mode: Өгөгдлийг *_battery.csv файлд хадгална." << endl;
    } else {
        cout << "Plugged-in mode: Өгөгдлийг стандарт .csv файлд хадгална." << endl;
    }
    
    for (int i=0; i<array_size; i++){
        array[i]=1.0;
    }
    
    int choise_method;
    cout << "Choise method" << endl << "1. Sequential" << endl << "2. OpenMP" << endl << "3. std::threads" << endl;
    cin >> choise_method;
    
    if(choise_method == 1){
        compute_sequential_function(10, num_thread, array_size, array, is_battery);
        string result_file = is_battery ? "csv/result_sequential_sin_battery.csv" : "csv/result_sequential_sin.csv";
        cout << "Sequential compute complete. Saved " << result_file << endl;
    } else if(choise_method == 2){
        compute_openmp_function(10, num_thread, array_size, array, is_battery);
        string result_file = is_battery ? "csv/result_openmp_sin_battery.csv" : "csv/result_openmp_sin.csv";
        cout << "OpenMP compute complete. Saved " << result_file << endl;
    } else if(choise_method == 3){
        compute_threads_function(10, num_thread, array_size, array, is_battery);
        string result_file = is_battery ? "csv/result_threads_sin_battery.csv" : "csv/result_threads_sin.csv";
        cout << "std::threads compute complete. Saved " << result_file << endl;
    }
    
    delete[] array;
    return 0;
}