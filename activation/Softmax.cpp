#include "Softmax.h"

vector<double> Softmax(vector<double> input)
{
	double maxExponent = MAX_EXPONENT;
	double minExponent = -MAX_EXPONENT;
	double K = maxExponent;
	for (size_t i = 0; i< input.size(); ++i)
	{
		if (input[i] > K)
			K = input[i];
		if (input[i] < minExponent)
			input[i] = minExponent;
	}

	double z = 0;
	vector<double>ps;

	if (K == maxExponent)
		K = 0;

	for (size_t i = 0; i< input.size(); ++i)
		z += exp(input[i] - K);
	for (size_t i = 0; i< input.size(); ++i)
		ps.push_back(exp(input[i] - K) / z);

	return ps;
}

vector<double> SoftmaxDerivative(vector<double> output)
{
	vector<double> derivatives;

	for (auto & o : output)
	{
		derivatives.push_back(o*(1 - o));
	}

	return derivatives;
}

