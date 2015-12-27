#ifndef TRAININGSET_H
#define TRAININGSET_H

#include <QString>
#include <vector>

using namespace std;

class DataSet
{
public:
    DataSet();
    virtual ~DataSet();

    vector<double> & Input(size_t set);
    vector<double> & Output(size_t set);
    size_t Size();

protected:
    vector<pair<vector<double>, vector<double>>> _set;
};

#endif // TRAININGSET_H
