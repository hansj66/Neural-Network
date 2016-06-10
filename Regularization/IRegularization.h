#ifndef IREGULARIZATION_H
#define IREGULARIZATION_H

class IRegularization
{
public:
	IRegularization() {}
	virtual double Cost() = 0;
	virtual double WeightUpdate(double & weightValue) = 0;
};

#endif
