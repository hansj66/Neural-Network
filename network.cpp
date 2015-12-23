#include "network.h"
#include <iostream>
#include <random>
#include <time.h>
#include <QFile>
#include <QTextStream>

// Reading materials:
//  http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
//  http://neuralnetworksanddeeplearning.com/chap2.html
//  http://neuralnetworksanddeeplearning.com/chap1.html


using namespace std;
Network::Network(Layers layers) :
	_layers(layers)
{
	random_device random;
	mt19937 engine(random());
	uniform_real_distribution<> distribution(-0.9, 0.9);

	for( auto l : _layers )
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
		_weights.push_back(Matrix(_activations.at(l).size(), _activations.at(l+1).size(), engine, distribution));
	}
}

Network::~Network()
{
}

inline double Network::Sigmoid(double x,double temperature)
{
	return (1.0 / (1+exp(-temperature*x)));
}

inline double Network::Derivative(double x)
{
	return Sigmoid(x)*(1-Sigmoid(x));
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

	for (size_t l=_layers.size()-2; l>0; l--)
	{
		for (size_t i=0; i<_layers.at(l); i++)
		{
			double x=0;
			for (size_t j=0; j<_layers.at(l+1); j++)
			{
				x+= _weights.at(l).Element(i, j) * _deltas.at(l+1).at(j) * _derivatives.at(l).at(i);
			}
			_deltas.at(l).at(i) = x;
		}
	}
}

void Network::UpdateWeights(double learningConstant)
{
	// Update weights
	for (size_t l=0; l<_layers.size()-1; l++)
	{
		for (size_t i=0; i<_layers.at(l); i++)
		{
			for (size_t j=0; j<_layers.at(l+1); j++)
			{
				_weights.at(l).Element(i, j) -= learningConstant * _deltas.at(l+1).at(j) * _activations.at(l).at(i);
			}
		}
		for (size_t j=0; j<_layers.at(l); j++)
		{
			_bias.at(l).at(j) -= learningConstant*_deltas.at(l).at(j);
		}
	}
}


vector<double> & Network::Activate(vector<double> input)
{
	if (input.size() != _layers.at(0))
	{
		cout << "Woops, we have discovered a slight impedance mismatch between the input layer and actual input.\nPanicking...\n";
		exit(1);
	}

	_activations.at(0) = Normalize(input);

	for (size_t l=1; l<_layers.size(); l++)
	{
		for (size_t j=0; j<_layers.at(l); j++)
		{
			double x=0;
			for (size_t i=0; i<_layers.at(l-1); i++)
			{
				x+= _activations.at(l-1).at(i)*_weights.at(l-1).Element(i, j) + _bias.at(l-1).at(i);
			}
			_activations.at(l).at(j) = Sigmoid(x);
			_derivatives.at(l).at(j) = Derivative(x);
		}
	}

	return _activations.at(_layers.size()-1);
}


double Network::SetError(vector<double> expected)
{
	double error = 0;
	size_t outputLayer = _layers.size()-1;
	for (size_t i=0; i<_activations.at(outputLayer).size(); i++)
	{
		double arg = expected.at(i)-_activations.at(outputLayer).at(i);
		error += 0.5*arg*arg;
		_deltas.at(outputLayer).at(i) = _activations.at(outputLayer).at(i) - expected.at(i);
	}

	return error;
}

void Network::ShowOff(TrainingSet & set)
{
	for (int i=0; i<set.Size(); i++)
	{
		auto output = Activate(set.Input(i));

		for (int j=0; j<set.Input(i).size(); j++)
			cout << set.Input(i).at(j) << " ";
		cout << " -> ";
		for (int j=0; j<output.size(); j++)
			cout << output.at(j) << " ";
		cout << endl;
	}
}


int Network::Train(TrainingSet & set, double learningConstant, double maxError, int maxIter)
{
	vector<double> errors;
	errors.resize(set.Size());
	for (int iter=0; iter<maxIter; iter++)
	{
		errors.clear();
		for (int i=0; i<set.Size(); i++)
		{

			double totalError = 0;
			Activate(set.Input(i));
			errors.push_back(SetError(set.Output(i)));
			BackPropagate();
			UpdateWeights(learningConstant);

			if (iter % 10000 == 0)
			{
				for (auto e: errors)
					totalError += e;
				std::cout << "Iteration: " << iter <<  " (set " << i << ") - error: " << totalError << std::endl;
			}

		}




		bool passed = true;
		for (auto e: errors)
			if (e > maxError)
				   passed = false;
		if (passed)
		{
			ShowOff(set);
			return iter;
		}

	}
	return -1;
}




void Network::TraceLayers()
{
	cout << "Layering structure : ";
	for( auto l : _layers )
	   cout << l << " ";
	cout << "\n" << endl ;
}

void Network::ExportAsDigraph(QString graphVizFileName)
{
	QFile file( graphVizFileName );
	if ( !file.open(QIODevice::ReadWrite) )
	{
		std::cout << "Error opening '" << graphVizFileName.toStdString() << "' for writing.";
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
		for (size_t i=0; i<_layers.at(l); i++)
		{
			for (size_t j=0; j<_layers.at(l+1); j++)
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


void Network::TraceLayerAttributes(string name, vector<vector<double>> & attribute)
{
	cout << name << ":\t";

	for (size_t l=0; l<attribute.size(); l++)
	{
		cout << "(";
		for (size_t a=0; a < attribute.at(l).size(); a++)
		{
			cout << attribute.at(l).at(a);
			if (a < attribute.at(l).size()-1)
				cout << " ";
		}
		cout << ")\t";
	}
	cout << endl;
}


void Network::Trace()
{
	cout << "--------------------------------------------------------------\n";
	cout << std::scientific;
	TraceLayers();
	TraceLayerAttributes("Activations", _activations);
	TraceLayerAttributes("Derivatives", _derivatives);
	TraceLayerAttributes("Deltas     ", _deltas);
	TraceLayerAttributes("Bias       ", _bias);

	cout << "\n" << _weights.size() << " weight matrixes defined\n";

	string indexNames = "ijklmnopqrstuvwxyz";
	string::iterator index = indexNames.begin();
	for (auto & w: _weights)
	{
		w.Trace("matrix", *index, *(index+1));
		index += 2;
		if (index == indexNames.end())
		{
			std::cout << "Please don't...";
			exit(1);
		}
	}

}



