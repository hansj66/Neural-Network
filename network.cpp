#include <iostream>
#include <limits>
#include "L2.h"
#include "network.h"
#include <QDataStream>
#include <QDebug>
#include <QFile>
#include <QTime>
#include <random>
#include "softmax.h"
#include "Tanh.h"
#include <time.h>

using namespace std;
Network::Network(Layers layers) :
	_layers(layers)
{
	random_device random;
	mt19937 engine(random());

	for( auto l : _layers )
	{
		auto a = vector<double>();
		a.resize(l);
		_activations.push_back(a);
		_derivatives.push_back(a);
		_deltas.push_back(a);

		auto bias = vector<double>();
		for (int b=0; b<l; b++)
		{
			bias.push_back(0);
		}
		_bias.push_back(bias);
	}

	for (size_t l=0; l<layers.size()-1; l++)
	{
		size_t i = _activations[l+1].size();
		size_t j = _activations[l].size();

		uniform_real_distribution<> distribution(-0.01/(sqrt(i)), 0.01/(sqrt(i))); // Tanh initializer
		//uniform_real_distribution<> distribution(0, 0.001); // ReLU initializer 

		_weights.push_back(Matrix(i, j, engine, distribution));

		_nesterovMomentum.push_back(Matrix(i, j));
	}
}

Network::~Network()
{
}

bool Network::operator != (const Network & rhs) const
{
	return !(*this == rhs);
}


bool Network::operator == (const Network & rhs) const
{
	if (_layers != rhs._layers)
		return false;
	if (_activations != rhs._activations)
		return false;
	if (_derivatives != rhs._derivatives)
		return false;
	if (_deltas != rhs._deltas)
		return false;
	if (_bias != rhs._bias)
		return false;
	if (_weights != rhs._weights)
		return false;

	return true;
}

bool Network::IsTopologicallyEquivalent(const Network & other) const
{
	if (_layers != other._layers)
		return false;

	return true;
}

vector<double> Network::Normalize(vector<double> input)
{
	vector<double> normalized;
	double max = input[0];
	for (size_t i=1; i<input.size(); i++)
		if (input[i] > max)
			max = input[i];

	for (size_t i=0; i<input.size(); i++)
		normalized.push_back(0 == max ? 0 : input[i] /= max);

	return normalized;
}

void Network::BackPropagateDeltas()
{
	// Error delta should be present at output before this is called - otherwise we're in deep dodo.

	size_t layerIndex = _layers.size() - 2;

	do
	{
		auto & m = _weights[layerIndex].AsVector();

		size_t iSize = _layers[layerIndex + 1];
		size_t jSize = _layers[layerIndex];

		size_t weightIndex = 0;
		for (size_t j = 0; j < jSize; j++)
		{
			double targetDelta = 0;
			for (size_t i = 0; i < iSize; i++)
			{
				targetDelta += m[weightIndex] * _deltas[layerIndex + 1][i] * _derivatives[layerIndex][j];
				weightIndex++;
			}
			_deltas[layerIndex][j] = targetDelta;
		}

		layerIndex--;

	} while (layerIndex >= 1);
}



void Network::UpdateWeights(double learningRate, double momentum, IRegularization * regularization)
{
	for (size_t l = 0; l < _layers.size() - 1; l++)
	{
		auto & w = _weights[l].AsVector();
		auto & n = _nesterovMomentum[l].AsVector();

		size_t index = 0;
		double nesterovNext = 0;
		double weightChange = 0;
		size_t iSize = _layers[l+1];
		size_t jSize = _layers[l];
		for (size_t i = 0; i < iSize; i++)
		{
			// Update weight matrix
			for (size_t j = 0; j < jSize; j++)
			{
				// Using Nesterov momentum
				index = j*iSize + i;
				nesterovNext = momentum*n[index] + learningRate  * _deltas[l + 1][i] * _activations[l][j];
				weightChange = momentum*n[index] - (1 + momentum) * nesterovNext - regularization->WeightUpdate(w[index]);
				w[index] += weightChange;
				n[index] = nesterovNext;

				// Weight update (without momentum)
				//index = j*iSize + i;
				//w[index] -= learningRate  * _deltas[l + 1][i] * _activations[l][j] - regularization->WeightUpdate(w[index]);
			}
			// Update bias vectors
			_bias[l+1][i] -= learningRate*_deltas[l+1][i];
		}
	}
}



vector<double> & Network::Activate(vector<double> input)
{
	if (input.size() != _layers[0])
	{
		cout << "Woops, we have discovered a slight impedance mismatch between the input layer and actual input.\nPanicking...\n";
		exit(1);
	}

	_activations[0] = Normalize(input);

	size_t outputLayer = _layers.size()-1;

	// ReLU activation of everything up to the output layer
	for (size_t activationLayer=1; activationLayer<_layers.size(); activationLayer++)
	{
		size_t feedLayer = activationLayer-1;
		auto & m = _weights[feedLayer].AsVector();

		size_t iSize = _layers[activationLayer];
		for (size_t i = 0; i < iSize; i++)
		{
			double sum = 0;
			auto & activations = _activations[feedLayer];
			for (size_t j = 0; j < _layers[feedLayer]; j++)
			{
				sum += activations[j] * m[j*iSize + i];
			}
			if (activationLayer != outputLayer)
			{
				_activations[activationLayer][i] = Tanh(sum + _bias[activationLayer][i]);
				_derivatives[activationLayer][i] = TanhDerivative(sum);
			}
			else
			{
				_activations[activationLayer][i] = sum;

			}
		}
	}

	_activations[outputLayer] = Softmax(_activations[outputLayer]);
	_derivatives[outputLayer] = SoftmaxDerivative(_activations[outputLayer]);


	for (size_t j=0; j<_layers[outputLayer]; j++)
	{
		VerifyFinite(_derivatives[outputLayer][j]);
	}
	return _activations[outputLayer];
}

double Network::SetError(vector<double> expected)
{
	// TODO: Parametrize L1 / L2 regularization ?

	size_t outputLayer = _layers.size()-1;

	double totalError = 0;
	for (size_t i=0; i<_activations[outputLayer].size(); i++)
	{
		//_deltas[outputLayer][i] = 0.5*(_activations[outputLayer][i]-expected[i])*(_activations[outputLayer][i] - expected[i]);
		_deltas[outputLayer][i] = _activations[outputLayer][i] - expected[i];
		totalError += 0.5*(_activations[outputLayer][i] - expected[i])*(_activations[outputLayer][i] - expected[i]);
	}
	return totalError;
}

void Network::ShowOff(DataSet & set)
{
	for (int i=0; i<set.Size(); i++)
	{
		cout << "Set: " << i << " - ";
		auto output = Activate(set.Input(i));
		cout << "label: ";
		for (auto o: set.Output(i))
			cout << o << " ";
		cout << " -> ";
		for (int j=0; j<output.size(); j++)
			cout << output[j] << "  ";
		cout << endl;
	}
}

inline int Network::Max(vector<double> & a)
{
	int maxIndex = 0;
	double maxValue = a[0];
	for (int i=1; i<a.size(); i++)
	{
		if (a[i] > maxValue)
		{
			maxValue = a[i];
			maxIndex = i;
		}
	}

	return maxIndex;
}

inline bool Network::ClassifiesAsEqual(vector<double> & a, vector<double> & b)
{
	return (Max(a) == Max(b));
}

int Network::Run(DataSet & set, string label, bool showOff)
{
	int correct = 0;
	for (int i = 0; i < set.Size(); i++)
	{
		Activate(set.Input(i));
		if (ClassifiesAsEqual(set.Output(i), _activations[_layers.size() - 1]))
		{
			correct++;
		}
	}
	if (showOff)
		ShowOff(set);

	cout << label  << ", Correct predictions: " << correct << "/" << set.Size() << endl;
	return correct;
}

int Network::Train(DataSet & set, double learningRate, double momentum, int maxEpoch, double maxError)
{
	QTime time;
	time.start();

	auto regularization = L2(_weights, set.Size(), 0.1 /* lambda*/ , learningRate);

	int epoch = 0;
	for (epoch=0; epoch<maxEpoch; epoch++)
	{

		double error = 0;
		for (int i = 0; i < set.Size(); i++)
		{
			Activate(set.Input(i));
			error += SetError(set.Output(i));
			BackPropagateDeltas();
			UpdateWeights(learningRate, momentum, &regularization);
		}
		cout.precision(8);
		cout << "Epoch: " << epoch <<  ". Total error: " << error << " (regularization cost: " << regularization.Cost() << ")" << endl;

		if (error < maxError)
		{
			cout << "Set trained in : " << time.elapsed()/1000.0 << " seconds." << endl;
			return epoch;
		}
	}

	return epoch;
}

template <typename T> void Network::Serialize(QDataStream & stream, vector<T> & state)
{
	stream << state.size();
	for (auto s: state)
		stream << s;
}

void Network::Serialize(string outputFileName)
{
	QFile file( QString::fromStdString(outputFileName) );
	if ( !file.open(QIODevice::ReadWrite) )
	{
		cout << "Error opening '" << outputFileName << "' for writing.";
		return;
	}

	QDataStream stream( &file );
	Serialize(stream, _layers);
	for (size_t l=0; l<_layers.size(); l++)
	{
		Serialize(stream, _activations[l]);
		Serialize(stream, _derivatives[l]);
		Serialize(stream, _deltas[l]);
		Serialize(stream, _bias[l]);
	}
	for (size_t w=0; w<_weights.size(); w++)
	{
		stream << _weights[w].I();
		stream << _weights[w].J();

		Serialize(stream, _weights[w].AsVector());
	}
}

template <typename T> void Network::Deserialize(QDataStream & stream, vector<T> & state)
{
	size_t size;
	stream >> size;
	state.resize(size);

	for (size_t i=0; i<size; i++)
	{
		T element;
		stream >> element;
		state[i] = element;
	}
}

void Network::InitializeNetwork(size_t s)
{
	_activations.resize(s);
	_derivatives.resize(s);
	_deltas.resize(s);
	_bias.resize(s);
	_weights.resize(s-1);
}

void Network::DeSerialize(string inputFileName)
{
	QFile file( QString::fromStdString(inputFileName) );
	if ( !file.open(QIODevice::ReadOnly) )
	{
		cout << "Error opening '" << inputFileName << "' for reading.";
		return;
	}

	QDataStream stream(file.readAll());

	Deserialize(stream, _layers);
	InitializeNetwork(_layers.size());
	for (size_t l=0; l<_layers.size(); l++)
	{
		Deserialize(stream, _activations[l]);
		Deserialize(stream, _derivatives[l]);
		Deserialize(stream, _deltas[l]);
		Deserialize(stream, _bias[l]);
	}
	for (size_t l=0; l<_layers.size()-1; l++)
	{
		size_t I;
		size_t J;
		stream >> I;
		stream >> J;
		Matrix m(I, J);
		Deserialize(stream, m.AsVector());
		_weights[l] = m;
	}
}



void Network::ExportAsDigraph(string graphVizFileName)
{
	QFile file( QString::fromStdString(graphVizFileName) );
	if ( !file.open(QIODevice::ReadWrite) )
	{
		cout << "Error opening '" << graphVizFileName << "' for writing.";
		return;
	}

	QTextStream stream( &file );

	stream << "digraph network\n{\n";
	stream << "node [margin=0   shape=circle style=filled];\n";
	stream << "rankdir = LR;\n";
	stream << "splines=line;\n";
	stream << "graph [ordering=\"out\"];";

	for (size_t l=0; l<_layers.size()-1; l++)
	{
		for (size_t i=0; i<_layers[l]; i++)
		{
			for (size_t j=0; j<_layers[l+1]; j++)
			{
				stream <<
				QString("L%1_N%2").arg(l).arg(i) <<
				" -> " <<
				QString("L%1_N%2").arg(l+1).arg(j) <<
				" [dir=none];\n";
			}
		}
	}
	stream << "\n}\n";
}

inline void Network::VerifyFinite(double value)
{
	if (!isfinite(value))
	{
		throw string("Runaway weights...");
	}
}




