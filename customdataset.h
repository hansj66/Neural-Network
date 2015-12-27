#ifndef CUSTOMTRAININGSET_H
#define CUSTOMTRAININGSET_H

#include "dataset.h"
#include <QString>

class CustomDataSet : public DataSet
{
public:
    CustomDataSet(QString trainingSet);
    virtual ~CustomDataSet();
};

#endif // CUSTOMTRAININGSET_H
