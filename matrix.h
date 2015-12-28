#ifndef MATRIX_H
#define MATRIX_H

#include <string>
#include <vector>
#include <random>
#include <QTextStream>

using namespace std;

class Matrix
{
public:
	Matrix(size_t i, size_t j,  mt19937 & engine, uniform_real_distribution<> & distribution);
	Matrix(size_t i, size_t j);
	~Matrix();
	double & Element(size_t i, size_t j);
	void Serialize(QTextStream & stream);

private:
	size_t	_i;
	size_t	_j;
	vector<double> _matrix;
};

#endif // MATRIX_H
