#include<iostream>
#include<fstream>
#include<vector>
#include<chrono>
#include<thread>
using namespace std;
using namespace std::chrono;

void transpose(double *b, double *b_t, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            b_t[j*n + i] = b[i*n + j];
        }
    }
}

void compute_sequential(double *a, double *b_t, double *c, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            double sum = 0.0;
            for(int k = 0; k < n; k++){
                sum += a[i*n + k] * b_t[j*n + k]; 
            }
            c[i*n + j] = sum;
        }
    }
}

void compute_openmp(double *a, double *b_t, double *c, int n, int num_threads){
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            double sum = 0.0;
            for(int k = 0; k < n; k++){
                sum += a[i*n + k] * b_t[j*n + k]; 
            }
            c[i*n + j] = sum;
        }
    }
}

void compute_thread_rows(double *a, double *b_t, double *c, int n, int start_row, int end_row){
    for(int i = start_row; i < end_row; i++){
        for(int j = 0; j < n; j++){
            double sum = 0.0;
            for(int k = 0; k < n; k++){
                sum += a[i*n + k] * b_t[j*n + k];
            }
            c[i*n + j] = sum;
        }
    }
}

void compute_threads(double *a, double *b_t, double *c, int n, int num_threads){
    vector<thread> threads;
    int rows_per_thread = n / num_threads;
    
    for(int i = 0; i < num_threads; i++){
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? n : (i + 1) * rows_per_thread;
        threads.push_back(thread(compute_thread_rows, a, b_t, c, n, start_row, end_row)); 
    }
    
    for(auto &t : threads){
        t.join();
    }
}

void compute_sequential_function(int sum_run, int max_threads, int n, double *a, double *b, double *c, bool is_battery){
    string file_path = is_battery ? "./csv/result_sequential_matrix_battery.csv" : "./csv/result_sequential_matrix.csv";
    ofstream file(file_path);
    file << "workload,implementation,threads,matrix_size,run,elapsed_ms" << endl;

    double *b_t = new double[n * n];
    transpose(b, b_t, n);

    for(int run_i = 1; run_i <= sum_run; run_i++){
        auto t1 = steady_clock::now();
        compute_sequential(a, b_t, c, n);
        auto t2 = steady_clock::now();
        auto duration = duration_cast<milliseconds>(t2 - t1);
        file << "matrix_multiply,sequential,1," << n << "," << run_i << "," << duration.count() << endl;
    }
    
    delete[] b_t;
    file.close();
}

void compute_openmp_function(int sum_run, int max_threads, int n, double *a, double *b, double *c, bool is_battery){
    string file_path = is_battery ? "./csv/result_openmp_matrix_battery.csv" : "./csv/result_openmp_matrix.csv";
    ofstream file(file_path);
    file << "workload,implementation,threads,matrix_size,run,elapsed_ms" << endl;

    double *b_t = new double[n * n];
    transpose(b, b_t, n);

    for(int run_i = 1; run_i <= sum_run; run_i++){
        auto t1 = steady_clock::now();
        compute_openmp(a, b_t, c, n, 1);
        auto t2 = steady_clock::now();
        auto duration = duration_cast<milliseconds>(t2 - t1);
        file << "matrix_multiply,openmp,1," << n << "," << run_i << "," << duration.count() << endl;

        for(int thread = 2; thread <= max_threads; thread += 2){
            auto t3 = steady_clock::now();
            compute_openmp(a, b_t, c, n, thread);
            auto t4 = steady_clock::now();
            auto duration_parallel = duration_cast<milliseconds>(t4 - t3);
            file << "matrix_multiply,openmp," << thread << "," << n << "," << run_i << "," << duration_parallel.count() << endl;
        }
    }
    
    delete[] b_t;
    file.close();
}   

void compute_threads_function(int sum_run, int max_threads, int n, double *a, double *b, double *c, bool is_battery){
    string file_path = is_battery ? "./csv/result_threads_matrix_battery.csv" : "./csv/result_threads_matrix.csv";
    ofstream file(file_path);
    file << "workload,implementation,threads,matrix_size,run,elapsed_ms" << endl;

    double *b_t = new double[n * n];
    transpose(b, b_t, n);

    for(int run_i = 1; run_i <= sum_run; run_i++){
        auto t1 = steady_clock::now();
        compute_threads(a, b_t, c, n, 1);
        auto t2 = steady_clock::now();
        auto duration = duration_cast<milliseconds>(t2 - t1);
        file << "matrix_multiply,threads,1," << n << "," << run_i << "," << duration.count() << endl;

        for(int thread = 2; thread <= max_threads; thread += 2){
            auto t3 = steady_clock::now();
            compute_threads(a, b_t, c, n, thread);
            auto t4 = steady_clock::now();
            auto duration_parallel = duration_cast<milliseconds>(t4 - t3);
            file << "matrix_multiply,threads," << thread << "," << n << "," << run_i << "," << duration_parallel.count() << endl;
        }
    }
    
    delete[] b_t;
    file.close();
}

int main(){
    int n;
    cout << "Enter matrix size (n for n×n): ";
    cin >> n;
    
    double *a_matrix = new double[n * n];
    double *b_matrix = new double[n * n];
    double *c_matrix = new double[n * n];
    
    for(int i = 0; i < n * n; i++){
        a_matrix[i] = 1.0;
        b_matrix[i] = 1.0;
        c_matrix[i] = 0.0;
    }
    
    int num_thread;
    cout << "Write max num threads: ";
    cin >> num_thread;
    cout << "Max threads: " << num_thread << endl;
    
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
    
    int choice_method;
    cout << "Choose method" << endl 
         << "1. Sequential" << endl 
         << "2. OpenMP" << endl 
         << "3. std::threads" << endl;
    cin >> choice_method;
    
    if(choice_method == 1){
        compute_sequential_function(10, num_thread, n, a_matrix, b_matrix, c_matrix, is_battery);
        string result_file = is_battery ? "csv/result_sequential_matrix_battery.csv" : "csv/result_sequential_matrix.csv";
        cout << "Sequential compute complete. Saved " << result_file << endl;
    } else if(choice_method == 2){
        compute_openmp_function(10, num_thread, n, a_matrix, b_matrix, c_matrix, is_battery);
        string result_file = is_battery ? "csv/result_openmp_matrix_battery.csv" : "csv/result_openmp_matrix.csv";
        cout << "OpenMP compute complete. Saved " << result_file << endl;
    } else if(choice_method == 3){
        compute_threads_function(10, num_thread, n, a_matrix, b_matrix, c_matrix, is_battery);
        string result_file = is_battery ? "csv/result_threads_matrix_battery.csv" : "csv/result_threads_matrix.csv";
        cout << "std::threads compute complete. Saved " << result_file << endl;
    }
    
    delete[] a_matrix;
    delete[] b_matrix;
    delete[] c_matrix;
    
    return 0;
}