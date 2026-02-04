#include <iostream>
#include<fstream>
#include<vector>
#include<chrono>
#include<thread>
using namespace std;
using namespace std::chrono;

int main(){
    int n;
    int m;
    cout << "Enter matrix size (n for n×n): ";
    cin >> n;
    cout << "Enter matrix size (m for m×m): ";
    cin >> m;
    cout <<"Matrix n x m:"<<n <<" x "<<m<<endl;
    vector<vector<double>> a(n, vector<double>(n));
    vector<vector<double>> b(m, vector<double>(m));
    vector<vector<double>> c(n, vector<double>(m));
    
    return 0;
}