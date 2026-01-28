#include<iostream>
#include<chrono>
#include<cmath>
#include<vector>
#include<thread>
using namespace std;

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
        int end = (i == chunk_size-1)? array_size : (i+1)*chunk_size;
        threads.push_back(thread(compute_thread, array, start, end));
    }
    for (auto &t : threads){
        t.join();
    }
}
int main (){
    int array_size = 80000000;
    vector<double> array(array_size, 1.0);
    int num_thread;
    cout << "Write num thread: ";
    cin >> num_thread;
    cout << "Num thread: " << num_thread << endl;
//--------------------OpenMP----------------------
    auto t1 = std::chrono::steady_clock::now();
    compute_openmp(array.data(), array_size, num_thread);
    auto t2 = std::chrono::steady_clock::now();
    auto d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
    std::cout.imbue(std::locale(""));
    cout << "Time taken by OpenMP compute " <<  d1.count() << " milliseconds " << endl;
    //--------------------thread------------------------
    auto t3 = std::chrono::steady_clock::now();
    compute_threads(array.data(), array_size, num_thread);
    auto t4 = std::chrono::steady_clock::now();
    auto d2 = std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3);
    std::cout.imbue(std::locale(""));
    cout << "Time taken by thread compute " <<  d2.count() << " milliseconds " << endl;
    //-------------------sequentail-------------------
    auto t5 = std::chrono::steady_clock::now();
    compute_sequential(array.data(), array_size);
    auto t6 = std::chrono::steady_clock::now();
    auto d3 = std::chrono::duration_cast<std::chrono::milliseconds>(t6-t5);
    std::cout.imbue(std::locale(""));
    cout << "Time taken by sequential compute " <<  d3.count() << " milliseconds " << endl;
    return 0;
}