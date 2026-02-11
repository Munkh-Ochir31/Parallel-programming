#include <iostream>
#include<fstream>
#include<vector>
#include<chrono>
#include<thread>
using namespace std;
using namespace std::chrono;

void transpose(double *b, double *b_t, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            b_t[j*n + i] = b[i*m + j];
        }
    }
}

void compute_sequential_no_transpose(double *a, double *b, double *c, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            double sum = 0.0;
            for(int k = 0; k < n; k++){
                sum += a[i*n + k] * b[k*m + j];
            }
            c[i*m + j] = sum;
        }
    }
}

void compute_openmp_no_transpose(double *a, double *b, double *c, int m, int n, int num_threads){
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            double sum = 0.0;
            for(int k = 0; k < n; k++){
                sum += a[i*n + k] * b[k*m + j];
            }
            c[i*m + j] = sum;
        }
    }
}

void compute_thread_rows_no_transpose(double *a, double *b, double *c, int m, int n, int start_row, int end_row){
    for(int i = start_row; i < end_row; i++){
        for(int j = 0; j < m; j++){
            double sum = 0.0;
            for(int k = 0; k < n; k++){
                sum += a[i*n + k] * b[k*m + j];
            }
            c[i*m + j] = sum;
        }
    }
}

void compute_threads_no_transpose(double *a, double *b, double *c, int m, int n, int num_threads){
    vector<thread> threads;
    int rows_per_thread = m / num_threads;
    
    for(int i = 0; i < num_threads; i++){
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? m : (i + 1) * rows_per_thread;
        threads.push_back(thread(compute_thread_rows_no_transpose, a, b, c, m, n, start_row, end_row)); 
    }
    
    for(auto &t : threads){
        t.join();
    }
}

void compute_sequential_with_transpose(double *a, double *b_t, double *c, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            double sum = 0.0;
            for(int k = 0; k < n; k++){
                sum += a[i*n + k] * b_t[j*n + k]; 
            }
            c[i*m + j] = sum;
        }
    }
}

void compute_openmp_with_transpose(double *a, double *b_t, double *c, int m, int n, int num_threads){
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            double sum = 0.0;
            for(int k = 0; k < n; k++){
                sum += a[i*n + k] * b_t[j*n + k]; 
            }
            c[i*m + j] = sum;
        }
    }
}

void compute_thread_rows_with_transpose(double *a, double *b_t, double *c, int m, int n, int start_row, int end_row){
    for(int i = start_row; i < end_row; i++){
        for(int j = 0; j < m; j++){
            double sum = 0.0;
            for(int k = 0; k < n; k++){
                sum += a[i*n + k] * b_t[j*n + k];
            }
            c[i*m + j] = sum;
        }
    }
}

void compute_threads_with_transpose(double *a, double *b_t, double *c, int m, int n, int num_threads){
    vector<thread> threads;
    int rows_per_thread = m / num_threads;
    
    for(int i = 0; i < num_threads; i++){
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? m : (i + 1) * rows_per_thread;
        threads.push_back(thread(compute_thread_rows_with_transpose, a, b_t, c, m, n, start_row, end_row)); 
    }
    
    for(auto &t : threads){
        t.join();
    }
}

void compute_sequential_no_transpose_function(int sum_run, int max_threads, int m, int n, double *a, double *b, double *c, bool is_battery){
    string file_path = is_battery ? "./csv/result_sequential_mxn_no_transpose_battery.csv" : "./csv/result_sequential_mxn_no_transpose.csv";
    ofstream file(file_path);
    file << "workload,implementation,threads,m_size,n_size,run,elapsed_ms" << endl;

    for(int run_i = 1; run_i <= sum_run; run_i++){
        auto t1 = steady_clock::now();
        compute_sequential_no_transpose(a, b, c, m, n);
        auto t2 = steady_clock::now();
        auto duration = duration_cast<milliseconds>(t2 - t1);
        file << "matrix_mxn_no_transpose,sequential,1," << m << "," << n << "," << run_i << "," << duration.count() << endl;
    }
    
    file.close();
}

void compute_openmp_no_transpose_function(int sum_run, int max_threads, int m, int n, double *a, double *b, double *c, bool is_battery){
    string file_path = is_battery ? "./csv/result_openmp_mxn_no_transpose_battery.csv" : "./csv/result_openmp_mxn_no_transpose.csv";
    ofstream file(file_path);
    file << "workload,implementation,threads,m_size,n_size,run,elapsed_ms" << endl;

    for(int run_i = 1; run_i <= sum_run; run_i++){
        auto t1 = steady_clock::now();
        compute_openmp_no_transpose(a, b, c, m, n, 1);
        auto t2 = steady_clock::now();
        auto duration = duration_cast<milliseconds>(t2 - t1);
        file << "matrix_mxn_no_transpose,openmp,1," << m << "," << n << "," << run_i << "," << duration.count() << endl;

        for(int thread = 2; thread <= max_threads; thread += 2){
            auto t3 = steady_clock::now();
            compute_openmp_no_transpose(a, b, c, m, n, thread);
            auto t4 = steady_clock::now();
            auto duration_parallel = duration_cast<milliseconds>(t4 - t3);
            file << "matrix_mxn_no_transpose,openmp," << thread << "," << m << "," << n << "," << run_i << "," << duration_parallel.count() << endl;
        }
    }
    
    file.close();
}   

void compute_threads_no_transpose_function(int sum_run, int max_threads, int m, int n, double *a, double *b, double *c, bool is_battery){
    string file_path = is_battery ? "./csv/result_threads_mxn_no_transpose_battery.csv" : "./csv/result_threads_mxn_no_transpose.csv";
    ofstream file(file_path);
    file << "workload,implementation,threads,m_size,n_size,run,elapsed_ms" << endl;

    for(int run_i = 1; run_i <= sum_run; run_i++){
        auto t1 = steady_clock::now();
        compute_threads_no_transpose(a, b, c, m, n, 1);
        auto t2 = steady_clock::now();
        auto duration = duration_cast<milliseconds>(t2 - t1);
        file << "matrix_mxn_no_transpose,threads,1," << m << "," << n << "," << run_i << "," << duration.count() << endl;

        for(int thread = 2; thread <= max_threads; thread += 2){
            auto t3 = steady_clock::now();
            compute_threads_no_transpose(a, b, c, m, n, thread);
            auto t4 = steady_clock::now();
            auto duration_parallel = duration_cast<milliseconds>(t4 - t3);
            file << "matrix_mxn_no_transpose,threads," << thread << "," << m << "," << n << "," << run_i << "," << duration_parallel.count() << endl;
        }
    }
    
    file.close();
}

void compute_sequential_with_transpose_function(int sum_run, int max_threads, int m, int n, double *a, double *b, double *c, bool is_battery){
    string file_path = is_battery ? "./csv/result_sequential_mxn_with_transpose_battery.csv" : "./csv/result_sequential_mxn_with_transpose.csv";
    ofstream file(file_path);
    file << "workload,implementation,threads,m_size,n_size,run,elapsed_ms" << endl;

    double *b_t = new double[n * m];
    transpose(b, b_t, n, m);

    for(int run_i = 1; run_i <= sum_run; run_i++){
        auto t1 = steady_clock::now();
        compute_sequential_with_transpose(a, b_t, c, m, n);
        auto t2 = steady_clock::now();
        auto duration = duration_cast<milliseconds>(t2 - t1);
        file << "matrix_mxn_with_transpose,sequential,1," << m << "," << n << "," << run_i << "," << duration.count() << endl;
    }
    
    delete[] b_t;
    file.close();
}

void compute_openmp_with_transpose_function(int sum_run, int max_threads, int m, int n, double *a, double *b, double *c, bool is_battery){
    string file_path = is_battery ? "./csv/result_openmp_mxn_with_transpose_battery.csv" : "./csv/result_openmp_mxn_with_transpose.csv";
    ofstream file(file_path);
    file << "workload,implementation,threads,m_size,n_size,run,elapsed_ms" << endl;

    double *b_t = new double[n * m];
    transpose(b, b_t, n, m);

    for(int run_i = 1; run_i <= sum_run; run_i++){
        auto t1 = steady_clock::now();
        compute_openmp_with_transpose(a, b_t, c, m, n, 1);
        auto t2 = steady_clock::now();
        auto duration = duration_cast<milliseconds>(t2 - t1);
        file << "matrix_mxn_with_transpose,openmp,1," << m << "," << n << "," << run_i << "," << duration.count() << endl;

        for(int thread = 2; thread <= max_threads; thread += 2){
            auto t3 = steady_clock::now();
            compute_openmp_with_transpose(a, b_t, c, m, n, thread);
            auto t4 = steady_clock::now();
            auto duration_parallel = duration_cast<milliseconds>(t4 - t3);
            file << "matrix_mxn_with_transpose,openmp," << thread << "," << m << "," << n << "," << run_i << "," << duration_parallel.count() << endl;
        }
    }
    
    delete[] b_t;
    file.close();
}   

void compute_threads_with_transpose_function(int sum_run, int max_threads, int m, int n, double *a, double *b, double *c, bool is_battery){
    string file_path = is_battery ? "./csv/result_threads_mxn_with_transpose_battery.csv" : "./csv/result_threads_mxn_with_transpose.csv";
    ofstream file(file_path);
    file << "workload,implementation,threads,m_size,n_size,run,elapsed_ms" << endl;

    double *b_t = new double[n * m];
    transpose(b, b_t, n, m);

    for(int run_i = 1; run_i <= sum_run; run_i++){
        auto t1 = steady_clock::now();
        compute_threads_with_transpose(a, b_t, c, m, n, 1);
        auto t2 = steady_clock::now();
        auto duration = duration_cast<milliseconds>(t2 - t1);
        file << "matrix_mxn_with_transpose,threads,1," << m << "," << n << "," << run_i << "," << duration.count() << endl;

        for(int thread = 2; thread <= max_threads; thread += 2){
            auto t3 = steady_clock::now();
            compute_threads_with_transpose(a, b_t, c, m, n, thread);
            auto t4 = steady_clock::now();
            auto duration_parallel = duration_cast<milliseconds>(t4 - t3);
            file << "matrix_mxn_with_transpose,threads," << thread << "," << m << "," << n << "," << run_i << "," << duration_parallel.count() << endl;
        }
    }
    
    delete[] b_t;
    file.close();
}

int main(){
    int m, n;
    cout << "Enter m (rows of A): ";
    cin >> m;
    cout << "Enter n (columns of A, rows of B): ";
    cin >> n;
    cout << "Matrix A: " << m << "×" << n << endl;
    cout << "Matrix B: " << n << "×" << m << endl;
    cout << "Result C: " << m << "×" << m << endl;
    
    double *a_matrix = new double[m * n];
    double *b_matrix = new double[n * m];
    double *c_matrix = new double[m * m];
    
    for(int i = 0; i < m * n; i++){
        a_matrix[i] = 1.0;
    }
    for(int i = 0; i < n * m; i++){
        b_matrix[i] = 1.0;
    }
    for(int i = 0; i < m * m; i++){
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
    
    int transpose_choice;
    cout << "\nB матрицийг эргүүлэх үү?" << endl
         << "1. Эргүүлэхгүй (No transpose - удаан)" << endl
         << "2. Эргүүлж (With transpose - хурдан)" << endl
         << "3. Хоёуланг нь (Both - харьцуулах)" << endl;
    cin >> transpose_choice;
    
    if(transpose_choice == 1 || transpose_choice == 3){
        cout << "\n===== ЭРГҮҮЛЭХГҮЙ ТООЦООЛОЛ (No Transpose) =====" << endl;
        if(choice_method == 1){
            compute_sequential_no_transpose_function(10, num_thread, m, n, a_matrix, b_matrix, c_matrix, is_battery);
            string result_file = is_battery ? "csv/result_sequential_mxn_no_transpose_battery.csv" : "csv/result_sequential_mxn_no_transpose.csv";
            cout << "Sequential (no transpose) complete. Saved " << result_file << endl;
        } else if(choice_method == 2){
            compute_openmp_no_transpose_function(10, num_thread, m, n, a_matrix, b_matrix, c_matrix, is_battery);
            string result_file = is_battery ? "csv/result_openmp_mxn_no_transpose_battery.csv" : "csv/result_openmp_mxn_no_transpose.csv";
            cout << "OpenMP (no transpose) complete. Saved " << result_file << endl;
        } else if(choice_method == 3){
            compute_threads_no_transpose_function(10, num_thread, m, n, a_matrix, b_matrix, c_matrix, is_battery);
            string result_file = is_battery ? "csv/result_threads_mxn_no_transpose_battery.csv" : "csv/result_threads_mxn_no_transpose.csv";
            cout << "std::threads (no transpose) complete. Saved " << result_file << endl;
        }
    }
    
    if(transpose_choice == 2 || transpose_choice == 3){
        cout << "\n===== ЭРГҮҮЛЖ ТООЦООЛОЛ (With Transpose) =====" << endl;
        if(choice_method == 1){
            compute_sequential_with_transpose_function(10, num_thread, m, n, a_matrix, b_matrix, c_matrix, is_battery);
            string result_file = is_battery ? "csv/result_sequential_mxn_with_transpose_battery.csv" : "csv/result_sequential_mxn_with_transpose.csv";
            cout << "Sequential (with transpose) complete. Saved " << result_file << endl;
        } else if(choice_method == 2){
            compute_openmp_with_transpose_function(10, num_thread, m, n, a_matrix, b_matrix, c_matrix, is_battery);
            string result_file = is_battery ? "csv/result_openmp_mxn_with_transpose_battery.csv" : "csv/result_openmp_mxn_with_transpose.csv";
            cout << "OpenMP (with transpose) complete. Saved " << result_file << endl;
        } else if(choice_method == 3){
            compute_threads_with_transpose_function(10, num_thread, m, n, a_matrix, b_matrix, c_matrix, is_battery);
            string result_file = is_battery ? "csv/result_threads_mxn_with_transpose_battery.csv" : "csv/result_threads_mxn_with_transpose.csv";
            cout << "std::threads (with transpose) complete. Saved " << result_file << endl;
        }
    }
    
    delete[] a_matrix;
    delete[] b_matrix;
    delete[] c_matrix;
    
    return 0;
}
