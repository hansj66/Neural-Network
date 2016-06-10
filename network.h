#ifndef NETWORK_H
#define NETWORK_H

#include <initializer_list>
#include "matrix.h"
#include <vector>
#include <string>
#include "dataset.h"
#include "IRegularization.h"

using namespace std;

typedef initializer_list<size_t> Layers;

class Network
{
public:
	Network(Layers layers);
	~Network();
	bool operator == (const Network & rhs) const;
	bool operator != (const Network & rhs) const;
	bool IsTopologicallyEquivalent(const Network & other) const;
	int Train(DataSet & set, double learningRate, double momentum, int maxEpoch, double maxError);
	int Run(DataSet & set, string label, bool showOff = false);

	void ExportAsDigraph(string graphVizFileName);
	void Serialize(string outputFileName);
	void DeSerialize(string inputFileName);

private:
	template <typename T> void Serialize(QDataStream & stream, vector<T> & state);
	template <typename T> void Deserialize(QDataStream & stream, vector<T> & state);
	void InitializeNetwork(size_t s);
	vector<size_t>_layers;
	vector<vector<double>> _activations;
	vector<vector<double>> _derivatives;
	vector<vector<double>> _deltas;
	vector<vector<double>> _bias;
	vector<Matrix> _weights;
	vector<Matrix> _nesterovMomentum;

	vector<double> & Activate(vector<double> input);
	void UpdateWeights(double learningRate, double momentum, IRegularization * regularization);
	void BackPropagateDeltas();
	std::vector<double> Normalize(std::vector<double> input);
	double SetError(vector<double> expected);

	void ShowOff(DataSet & set);
	inline bool ClassifiesAsEqual(vector<double> & a, vector<double> & b);
	inline int Max(vector<double> & a);
	inline void VerifyFinite(double value);
};

#endif // NETWORK_H
