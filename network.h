#ifndef NETWORK_H
#define NETWORK_H

#include <initializer_list>
#include "matrix.h"
#include <vector>
#include <string>
#include "dataset.h"

using namespace std;

typedef initializer_list<size_t> Layers;

class Network
{
public:
	Network(Layers layers);
	~Network();

	int Train(DataSet & set, double learningConstant, int maxEpoch);
	void Run(DataSet & set);

	void ExportAsDigraph(string graphVizFileName);
	void Serialize(string outputFileName);

private:
	vector<size_t> _nodes;
	vector<vector<double>> _activations;
	vector<vector<double>> _derivatives;
	vector<vector<double>> _deltas;
	vector<vector<double>> _bias;
	vector<Matrix> _weights;
	// vector<Matrix> _deltaWeights; For distributed sync
	// vector<double> _deltaBias

	vector<double> & Activate(vector<double> input);
	void UpdateWeights(double learningConstant);
	void BackPropagate();
	inline double Sigmoid(double x, double temperature = 1);
	inline double SigmoidDerivative(double x);
	inline vector<double> Softmax(vector<double> input);
	inline double SoftmaxDerivative(double x);

	std::vector<double> Normalize(std::vector<double> input);
	void SetError(vector<double> expected);

	void ShowOff(DataSet & set);
	bool IsEqual(vector<double> & a, vector<double> & b);
};

#endif // NETWORK_H
