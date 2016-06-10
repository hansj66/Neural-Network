#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <vector>

using namespace std;

const int MAX_EXPONENT = 700;
const int MAX_TANH_ARG = 350;

vector<double> Softmax(vector<double> input);
vector<double> SoftmaxDerivative(vector<double> output);



#endif