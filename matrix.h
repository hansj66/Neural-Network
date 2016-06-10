#ifndef MATRIX_H
#define MATRIX_H

#include <string>
#include <vector>
#include <random>
#include <QDataStream>

using namespace std;

class Matrix
{
public:
    Matrix(size_t i, size_t j,  mt19937 & engine, uniform_real_distribution<> & distribution);
    Matrix(size_t i=0, size_t j=0);
	bool operator == (const Matrix & rhs) const;
	bool operator != (const Matrix & rhs) const;
	~Matrix();
    void Serialize(QDataStream & stream);
	size_t I();
	size_t J();
	vector<double> & AsVector();
	void SetElement(size_t i, size_t j, double value);
	double & GetElement(size_t i, size_t j);
private:
	size_t	_i;
	size_t	_j;
	vector<double> _matrix;
};

#endif // MATRIX_H
