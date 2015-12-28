#include "network.h"
#include <iostream>
#include <random>
#include <time.h>
#include <QFile>
#include <QTextStream>
#include <limits>
#include <QTime>


using namespace std;
Network::Network(Layers layers) :
	_nodes(layers)
{
	random_device random;
	mt19937 engine(random());
	uniform_real_distribution<> distribution(-0.9, 0.9);

	for( auto l : _nodes )
	{
		auto a = vector<double>();
		a.resize(l);
		_activations.push_back(a);
		_derivatives.push_back(a);
		_deltas.push_back(a);

		auto bias = vector<double>();
		for (int biasNode=0; biasNode<l; biasNode++)
		{
			bias.push_back(distribution(engine));
		}
		_bias.push_back(bias);
	}

	for (size_t l=0; l<layers.size()-1; l++)
	{
		size_t i = _activations.at(l).size();
		size_t j = _activations.at(l+1).size();

		_weights.push_back(Matrix(i, j, engine, distribution));
//		_deltaWeights.push_back(Matrix(i, j)); For distributed sync
	}
}

Network::~Network()
{
}

inline double Network::Sigmoid(double x,double temperature)
{
	return (1.0 / (1+exp(-temperature*x)));
}

inline double Network::SigmoidDerivative(double x)
{
	return Sigmoid(x)*(1-Sigmoid(x));
}

inline vector<double> Network::Softmax(vector<double> input)
{
	double maxExponent = 700;
	double minExponent = -700;
	double max = maxExponent;
	double min = minExponent;
	for (size_t i=0; i< input.size(); ++i)
	{
		if (input.at(i) > max)
			max = input.at(i);
		else if (input.at(i) < min)
			min = input.at(i);
	}

	double z = 0;
	vector<double>ps;

	if ((max != maxExponent)  &&
		(min == minExponent))
	{
		for (size_t i=0; i< input.size(); ++i)
			 z += exp(input.at(i)-max);
		for (size_t i=0; i< input.size(); ++i)
			ps.push_back(exp(input.at(i)-max)/z);
	}
	else if ((max == maxExponent)  &&
			 (min != minExponent))
	{
		for (size_t i=0; i< input.size(); ++i)
			 z += exp(input.at(i)+abs(min));
		for (size_t i=0; i< input.size(); ++i)
			ps.push_back(exp(input.at(i)+abs(min))/z);
	}
	else if ((max != maxExponent)  &&
			 (min != minExponent))
	{
		Q_ASSERT(false); // We're basically fucked - until we think of something clever...

		cout << "\\n. This is a bit embarrassing, but we seem to have encountered a small issue, regarding the implementation of the softmax function. Aborting... Please call back later.\n\n";
	}
	else
	{
		for (size_t i=0; i< input.size(); ++i)
			 z += exp(input.at(i));

		// Still not entirely safe... ?

		Q_ASSERT(z != 0);
		Q_ASSERT(isfinite(z));

		for (size_t i=0; i< input.size(); ++i)
		{
			Q_ASSERT(isfinite(input.at(i)));
			ps.push_back(exp(input.at(i))/z);
		}
	}

	for (size_t i=0; i<ps.size(); i++)
	{
		Q_ASSERT(isfinite(ps.at(i)));
	}

	return ps;
}

inline double Network::SoftmaxDerivative(double x)
{
	return x * (1 - x);
}


std::vector<double> Network::Normalize(std::vector<double> input)
{
	// TODO: Normalize to mid /std.

	std::vector<double> normalized;
	double max = input.at(0);
	for (size_t i=1; i<input.size(); i++)
		if (input.at(i) > max)
			max = input.at(i);

	for (size_t i=0; i<input.size(); i++)
		normalized.push_back(0 == max ? 0 : input.at(i) /= max);

	return normalized;
}

void Network::BackPropagate()
{
	// Deltas should be present at output before this is called - otherwise we're in deep dodo.

	for (size_t l=_nodes.size()-2; l>0; l--)
	{
		for (size_t i=0; i<_nodes.at(l); i++)
		{
			double x=0;
			for (size_t j=0; j<_nodes.at(l+1); j++)
			{
				x+= _weights.at(l).Element(i, j) *
						_deltas.at(l+1).at(j) *
						_derivatives.at(l).at(i);
			}
			_deltas.at(l).at(i) = x;
		}
	}
}

void Network::UpdateWeights(double learningConstant)
{
	for (size_t l=0; l<_nodes.size()-1; l++)
	{
		// Update weight matrixes
		for (size_t i=0; i<_nodes.at(l); i++)
		{
			for (size_t j=0; j<_nodes.at(l+1); j++)
			{
				double deltaWeight =  learningConstant  * _deltas.at(l+1).at(j) * _activations.at(l).at(i);
// 				_deltaWeights.at(l).Element(i, j) = delta; For distributed sync
				_weights.at(l).Element(i, j) -= deltaWeight;
			}
			// Update biase vectors
			_bias.at(l).at(i) -= learningConstant*_deltas.at(l).at(i);
		}
	}

	// MQTT Mosquitto
	// Test mellom prosesser først med
	// Fyr opp en egen vektserver. Bruk TCP/IP
	// Ikke oppdater vektmatrise over,
	// men signaller at ferdig.
	// La master konsolidere alle vekter
	// Vent på synkronisering og hent vekter tilbake i en smell
}


vector<double> & Network::Activate(vector<double> input)
{
	if (input.size() != _nodes.at(0))
	{
		cout << "Woops, we have discovered a slight impedance mismatch between the input layer and actual input.\nPanicking...\n";
		exit(1);
	}

	_activations.at(0) = Normalize(input);

	size_t outputLayer = _nodes.size()-1;

	// Sigmoid activation of everything up to the output layer
	for (size_t activationLayer=1; activationLayer<_nodes.size(); activationLayer++)
	{
		size_t feedLayer = activationLayer-1;
		for (size_t j=0; j<_nodes.at(activationLayer); j++)
		{
			double sum=0;
			for (size_t i=0; i<_nodes.at(feedLayer); i++)
			{
				sum+= _activations.at(feedLayer).at(i) *
					  _weights.at(feedLayer).Element(i, j) +
					  _bias.at(feedLayer).at(i);
				Q_ASSERT(isfinite(sum));
			}
			if (activationLayer != outputLayer)
			{
				_activations.at(activationLayer).at(j) = Sigmoid(sum);
				_derivatives.at(activationLayer).at(j) = SigmoidDerivative(sum);
			}
			else
			{
				_activations.at(activationLayer).at(j) = sum;
				_derivatives.at(activationLayer).at(j) = SoftmaxDerivative(sum);
			}
		}
	}


	_activations.at(outputLayer) = Softmax(_activations.at(outputLayer));

	return _activations.at(outputLayer);
}



void Network::SetError(vector<double> expected)
{
	size_t outputLayer = _nodes.size()-1;
	for (size_t i=0; i<_activations.at(outputLayer).size(); i++)
	{
		_deltas.at(outputLayer).at(i) = _activations.at(outputLayer).at(i)-expected.at(i);
	}
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
			cout << output.at(j) << "  ";
		cout << endl;
	}
}

bool Network::IsEqual(vector<double> & a, vector<double> & b)
{
	for (int i=0; i<a.size(); i++)
	{
		if (abs(a.at(i) - abs(b.at(i))) > 0.01)
			return false;
	}
	return true;
}

void Network::Run(DataSet & set)
{
	size_t correct = 0;
	for (int i=0; i<set.Size(); i++)
	{
		Activate(set.Input(i));
		if (IsEqual(set.Output(i), _activations.at(_nodes.size()-1)))
			correct++;
	}
	ShowOff(set);

	std::cout << "Correct predictions: " << correct << "/" << set.Size() << std::endl;
}


int Network::Train(DataSet & set, double learningConstant, int maxEpoch)
{
	QTime time;
	int epoch = 0;
	for (epoch=0; epoch<maxEpoch; epoch++)
	{
		size_t correct = 0;
		for (int i=0; i<set.Size(); i++)
		{
// #ifdef _DEBUG
//			time.start();
//#endif

			Activate(set.Input(i));
			SetError(set.Output(i));
			if (IsEqual(set.Output(i), _activations.at(_nodes.size()-1)))
				correct++;
			BackPropagate();
			UpdateWeights(learningConstant);
// #ifdef _DEBUG
//			cout << time.elapsed() << endl;
// #endif
		}

		std::cout << "Epoch: " << epoch <<  ". Correct predictions: " << correct << "/" << set.Size() << std::endl;

		if (correct == set.Size())
		{
			return epoch;
		}
	}
	return epoch;
}



void Network::Serialize(string outputFileName)
{
	QFile file( QString::fromStdString(outputFileName) );
	if ( !file.open(QIODevice::ReadWrite) )
	{
		std::cout << "Error opening '" << outputFileName << "' for writing.";
		return;
	}

	QTextStream stream( &file );
	stream << "Network: ";
	for (auto l: _nodes)
		stream << l << " ";
	stream << endl;
	for (size_t l=0; l<_nodes.size(); l++)
	{
		for (size_t b=0; b<_nodes.at(l); b++)
		{
			stream << _bias.at(l).at(b) << " ";
		}
		stream << endl;
	}
	stream << endl;
	for (size_t l=0; l<_nodes.size()-1; l++)
		_weights.at(l).Serialize(stream);
	stream << endl;
}


void Network::ExportAsDigraph(string graphVizFileName)
{
	QFile file( QString::fromStdString(graphVizFileName) );
	if ( !file.open(QIODevice::ReadWrite) )
	{
		std::cout << "Error opening '" << graphVizFileName << "' for writing.";
		return;
	}

	QTextStream stream( &file );

	stream << "digraph network\n{\n";
	stream << "node [margin=0   shape=circle style=filled];\n";
	stream << "rankdir = LR;\n";
	stream << "splines=line;\n";
	stream << "graph [ordering=\"out\"];";

	for (size_t l=0; l<_nodes.size()-1; l++)
	{
		for (size_t i=0; i<_nodes.at(l); i++)
		{
			for (size_t j=0; j<_nodes.at(l+1); j++)
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





