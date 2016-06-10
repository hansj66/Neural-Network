class IActivation
{
public:
	IActivation() {}
	virtual double Function() = 0;
	virtual double Derivative() = 0;
};
