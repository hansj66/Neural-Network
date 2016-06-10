#include "ReLU.h"

double ReLU(double x)
{
	if (x > 0)
		return x;
	return 0;
}

double ReLUDerivative(double x)
{
	if (x<0)
		return 0;
	return 1;
}
