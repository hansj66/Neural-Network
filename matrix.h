#ifndef MATRIX_H
#define MATRIX_H

#include <string>
#include <vector>
#include <random>

using namespace std;

class Matrix
{
public:
    Matrix(size_t i, size_t j,  mt19937 & engine, uniform_real_distribution<> & distribution);
	~Matrix();
	double & Element(size_t i, size_t j);
	void Trace(string name, char nameI, char nameJ);

private:
	size_t	_i;
	size_t	_j;
	vector<vector<double>> _matrix;
};

#endif // MATRIX_H
