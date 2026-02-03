#include<iostream>
#include<fstream>

int main(){
    std::string folder_paph = "./csv";
    std::string csv_path = "./lab1/benchmark_results.csv";
    std::string line;
    std::fstream fout;
    std::ifstream file(csv_path);

    if(!file.is_open()){
        std::cerr << "File is empty " << std::endl;
    }

    std:getline(file, line);
    while(getline(file, line)){
        std::cout << line << "\n";
    }
    file.close();
}