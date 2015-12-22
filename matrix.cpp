#include <iostream>
#include "matrix.h"

Matrix::Matrix(size_t i, size_t j, mt19937 & engine, uniform_real_distribution<> & distribution) :
	_i(i),
	_j(j)
{
	for (int ii=0; ii<_i; ii++)
	{
		auto jj=vector<double>();
		for (size_t element=0; element<_j; element++)
            jj.push_back(distribution(engine));
		_matrix.push_back(jj);
	}
}

Matrix::~Matrix()
{
}

double & Matrix::Element(size_t i, size_t j)
{
	return _matrix.at(i).at(j);
}


void Matrix::Trace(string name, char nameI, char nameJ)
{
	cout << "\n" << name << "[" << nameI << ", " << nameJ << "]\n";

	for (int j=0; j<_j; j++)
	{
		for (int i=0; i<_i; i++)
		{
			cout << _matrix.at(i).at(j) << "\t";
		}
		cout << endl;
	}
}

