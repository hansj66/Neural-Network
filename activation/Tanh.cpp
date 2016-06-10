#include "Tanh.h"

using namespace std;

double Tanh(double x)
{
	if (x == -numeric_limits<double>::infinity())
		return -1;
	else if (x == numeric_limits<double>::infinity())
		return 1;
	double e2x = exp(2 * x);
	return (e2x - 1) / (e2x + 1);
}

double TanhDerivative(double x)
{
	double tanhX = Tanh(x);
	return (1 - tanhX*tanhX);
}

