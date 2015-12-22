#include "trainingset.h"
#include <iostream>
#include <QStringList>
#include <QString>
#include <QFile>
#include <QTextStream>

using namespace std;

TrainingSet::TrainingSet(QString trainingSet)
{
    QFile file(trainingSet);
    if(!file.open(QIODevice::ReadOnly))
    {
        cout << "Unable to open training set '" << trainingSet.toStdString() << "'\n";
        return;
    }

    QTextStream in(&file);

    while(!in.atEnd())
    {
        QString line = in.readLine().trimmed();
        if (line.isEmpty())
            continue;
        if (line.at(0) == "#")
            continue;

        QStringList sets = line.split("->", QString::SkipEmptyParts);
        if (sets.size() != 2)
        {
            cout << "Error. Unable to parse training set file. Offending line: '" << line.toStdString() << "'\n";
            return;
        }

        vector<double> input;
        QStringList inputSet = sets.at(0).split(",", QString::SkipEmptyParts);
        for (auto s: inputSet)
            input.push_back(s.toDouble());
        vector<double> output;
        QStringList outputSet = sets.at(1).split(",", QString::SkipEmptyParts);
        for (auto s: outputSet)
            output.push_back(s.toDouble());

        _set.push_back(make_pair(input, output));
    }

}

TrainingSet::~TrainingSet()
{
}

vector<double> & TrainingSet::Input(size_t set)
{
    return _set.at(set).first;
}

vector<double> & TrainingSet::Output(size_t set)
{
    return _set.at(set).second;
}

size_t TrainingSet::Size()
{
    return _set.size();
}




