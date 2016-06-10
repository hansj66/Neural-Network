#include "dataset.h"
#include <iostream>
#include <cstdlib>
#include <random>

using namespace std;

DataSet::DataSet() :
	_currentIndex(0)
{
}

DataSet::~DataSet()
{
}

vector<double> & DataSet::Input(size_t set)
{
	return _set[set].first;
}

vector<double> & DataSet::Output(size_t set)
{
	return _set[set].second;
}

size_t DataSet::Size()
{
	return _set.size();
}

DataSet DataSet::CreateBatch(size_t elements)
{
	DataSet batch;

	/*
	random_device random;
	mt19937 engine(random());
	uniform_real_distribution<> distribution(0, _set.size() - elements - 1);
	_currentIndex = static_cast<size_t>(distribution(engine));

	if (elements > _set.size())
	{
		cout << "Warning. Batch size is larger than set size. Defaulting to set size.\n";
		_currentIndex = 0;
		elements = _set.size();
	}
	*/

	if (_currentIndex + elements > _set.size())
    {
		elements = _set.size()-_currentIndex;
        _currentIndex = 0;
    }

	cout << "\nCreated new batch [" << _currentIndex << ", " << _currentIndex+elements-1 << "]\n";

	for (size_t e = _currentIndex; e<_currentIndex+elements; e++)
	{
		auto element = make_pair(_set[e].first, _set[e].second);
		batch._set.push_back(element);
	}

	if (_currentIndex + elements > _set.size())
		_currentIndex = 0;
	else
		_currentIndex += elements;
	return batch;
}





