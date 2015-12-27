#ifndef MNISTTRAININGSET_H
#define MNISTTRAININGSET_H

#include "dataset.h"
#include <string>

using namespace std;

class MNISTDataSet : public DataSet
{
public:
    MNISTDataSet(string input, string output, quint32 maxImages = -1);
    virtual ~MNISTDataSet();
    quint32 Parameter(unsigned char * memory);
};

#endif // MNISTTRAININGSET_H
