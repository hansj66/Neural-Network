#ifndef NETWORK_H
#define NETWORK_H

#include <initializer_list>
#include "matrix.h"
#include <vector>
#include <string>
#include "trainingset.h"

using namespace std;

typedef initializer_list<size_t> Layers;

class Network
{
public:
	Network(Layers layers);
	~Network();

    int Network::Train(TrainingSet & set, double learningConstant, double maxError, int maxIter);

	void Trace();

private:
	vector<size_t> _layers;
	vector<vector<double>> _activations;
	vector<vector<double>> _derivatives;
	vector<vector<double>> _deltas;
	vector<vector<double>> _bias;
	vector<Matrix> _weights;

    vector<double> & Activate(vector<double> input);
    void UpdateWeights(double learningConstant);
    void BackPropagate();
    inline double Sigmoid(double x, double temperature = 1);
	inline double Derivative(double x);
	std::vector<double> Normalize(std::vector<double> input);
    double SetError(vector<double> expected);
    void ShowOff(TrainingSet & set);

	// Remove in release
	void TraceLayers();
	void TraceLayerAttributes(string name, vector<vector<double>> & attribute);
	void TraceMatrixes();
};

#endif // NETWORK_H
