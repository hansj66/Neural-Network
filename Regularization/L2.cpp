#include "L2.h"

L2::L2(vector<Matrix> & matrixes, size_t setSize, double lambda, double learningRate) :
	_matrixes(matrixes),
	_setSize(setSize),
	_lambda(lambda),
	_learningRate(learningRate)
{
}

double L2::Cost()
{
	double cost = 0;
	for (auto & wm : _matrixes)
	{
		auto & m = wm.AsVector();
		for (auto & w : m)
		{
			cost += w*w;
		}
	}
	return _lambda*cost / (2 * _setSize);
}

double L2::WeightUpdate(double & weightValue)
{
	return _lambda*_learningRate*weightValue / _setSize;
}

