#include <iostream>
#include "matrix.h"

Matrix::Matrix(size_t i, size_t j, mt19937 & engine, uniform_real_distribution<> & distribution) :
	_i(i),
	_j(j)
{
	for (int ii=0; ii<_i; ii++)
	{
		for (size_t element=0; element<_j; element++)
			_matrix.push_back(distribution(engine));
	}
}

Matrix::Matrix(size_t i, size_t j) :
	_i(i),
	_j(j)
{
	for (int ii=0; ii<_i; ii++)
	{
		for (size_t element=0; element<_j; element++)
        {
			_matrix.push_back(0);
        }
	}
}


Matrix::~Matrix()
{
}


bool Matrix::operator == (const Matrix & rhs) const
{
	return _matrix == rhs._matrix;
}

bool Matrix::operator != (const Matrix & rhs) const
{
	return _matrix != rhs._matrix;
}


size_t Matrix::I()
{
	return _i;
}

size_t Matrix::J()
{
	return _j;
}

vector<double> & Matrix::AsVector()
{
	return _matrix;
}


void Matrix::Serialize(QDataStream & stream)
{
	stream << _i;
	stream << _j;

	for (size_t j=0; j<_j; j++)
	{
		for (size_t i=0; i<_i; i++)
		{
			stream << _matrix[j*_i+i];
		}
	}
}

void Matrix::SetElement(size_t i, size_t j, double value)
{
	_matrix[j*_i + i] = value;
}

double & Matrix::GetElement(size_t i, size_t j)
{
	return _matrix[j*_i+i];
}



