#include "Sigmoid.h"
#include <cmath>

double Sigmoid(double x, double temperature)
{
	return (1.0 / (1 + exp(-temperature*x)));
}

double SigmoidDerivative(double x)
{
	return Sigmoid(x)*(1 - Sigmoid(x));
}

