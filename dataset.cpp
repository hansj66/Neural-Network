#include "dataset.h"
#include <iostream>

using namespace std;

DataSet::DataSet()
{
}

DataSet::~DataSet()
{
}

vector<double> & DataSet::Input(size_t set)
{
    return _set.at(set).first;
}

vector<double> & DataSet::Output(size_t set)
{
    return _set.at(set).second;
}

size_t DataSet::Size()
{
    return _set.size();
}




