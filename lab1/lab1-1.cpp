#include<iostream>
#include<chrono>
#include<thread>
#include<vector>
using namespace std;
using namespace std::chrono;

double compute_sequential(double *array, int array_size){
    double sum = 0.0;
    for (int i=0; i< array_size; i++){
        sum +=array[i];
    }
    return sum;
}

double compute_openmp(double *array, int array_size, int num_thread){
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) num_threads(num_thread)
    for (int i= 0; i<array_size; i++){
        sum +=array[i];
    }
    return sum;
}
void compute_thread(double *array, int start, int end, double *result){
    double sum = 0.0;
    for(int i = start; i<end; i++){
        sum += array[i];
    }
    *result = sum;
}

double compute_threads(double *array, int array_size, int num_threads){
    vector<thread> threads;
    vector<double> treads_sum(num_threads, 0.0);
    int chunk_size = array_size / num_threads;
    for (int i=0; i< num_threads; i++){
        int start = i * chunk_size;
        int end = (i==chunk_size-1)? array_size : (i+1)*chunk_size;
        threads.push_back(thread(compute_thread, array, start, end, &treads_sum[i]));
    }
    for (auto &t : threads){
        t.join();
    }
    double sum = 0.0;
    for(double st : treads_sum ){
        sum += st;
    }
    return sum;
}

int main(){
    int array_size=80000000;
    double *array = new double[array_size];
    int num_thread;
    cout << "Write num thread: ";
    cin >> num_thread;
    cout << "Num thread: " << num_thread << endl;
    auto start = steady_clock::now();
    for (int i=0;i<array_size;i++){
        array[i]=1.0;
    }
    auto stop = steady_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout<<"Time taken by sequential loop: "<<duration.count()<<" milliseconds"<<endl;
    //--------------------OpenMP----------------------
    auto t1 = steady_clock::now();
    double sum_openmp = compute_openmp(array, array_size, num_thread);
    auto t2 = steady_clock::now();
    auto d1 = duration_cast<milliseconds>(t2-t1);
    std::cout.imbue(std::locale(""));
    cout << "Time taken by OpenMP compute " <<  d1.count() << " milliseconds " << endl;
    cout << "Count by OpenMP: " << sum_openmp << endl;
    //--------------------thread------------------------
    auto t3 = steady_clock::now();
    double sum_thread = compute_threads(array, array_size, num_thread);
    auto t4 = steady_clock::now();
    auto d2 = duration_cast<milliseconds>(t4-t3);
    std::cout.imbue(std::locale(""));
    cout << "Time taken by thread compute " <<  d2.count() << " milliseconds " << endl;
    cout << "Count by thread: " << sum_thread << endl;
    //-------------------sequentail-------------------
    auto t5 = steady_clock::now();
    double sum_sequantial = compute_sequential(array, array_size);
    auto t6 = steady_clock::now();
    auto d3 = duration_cast<milliseconds>(t6-t5);
    std::cout.imbue(std::locale(""));
    cout << "Time taken by sequential compute " <<  d3.count() << " milliseconds " << endl;
    cout << "Count by sequential: " << sum_sequantial << endl;  
    delete[] array;
    return 0;
}