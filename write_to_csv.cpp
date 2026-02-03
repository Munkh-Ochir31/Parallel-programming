#include<iostream>
#include<fstream>
int main(){
    std::string file_path = "test.csv";
    std::ofstream file(file_path);

    int id = 1;
    std::string name = "testName";
    bool result = false;

    file << id << "," << name << "," << result << std::endl;

    return 0;
}