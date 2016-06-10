#include "IRegularization.h"
#include "matrix.h"

class L2 : public IRegularization
{
public:
	L2(vector<Matrix> & matrixes, size_t setSize, double lambda, double learningRate);
	double Cost() override;
	double WeightUpdate(double & weightValue) override;

private:
	vector<Matrix> & _matrixes;
	size_t _setSize;
	double _lambda;
	double _learningRate;
};
