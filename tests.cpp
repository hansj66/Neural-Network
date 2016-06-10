#include "tests.h"
#include "network.h"
#include "customdataset.h"
#include <ctime>
#include <QFile>

namespace UnitTest
{

Tests::Tests()
{

}

Tests::~Tests()
{

}


void Tests::init()
{
	srand(time(0));
}

void Tests::cleanup()
{
}

void Tests::TestNetworkCompareTopology()
{
	Network n1({784, 128,  10});
	Network n2({784, 128});
	Network n3({784, 64,  10});
	Network n4({784, 128,  10});

	QVERIFY(n1 != n2);
	QVERIFY(n2 != n3);
	QVERIFY(n1 != n3);
	QVERIFY(n1.IsTopologicallyEquivalent(n4));
	QVERIFY(n4.IsTopologicallyEquivalent(n1));
	QVERIFY(!n1.IsTopologicallyEquivalent(n2));
	QVERIFY(!n1.IsTopologicallyEquivalent(n3));
}

void Tests::TestNetworkCompareState()
{
	Network n1({784, 128,  10});
	Network n2({784, 128,  10});
	QVERIFY(n1.IsTopologicallyEquivalent(n2));
	// Networks are initialized with random weights when created
	// Topologically identical networks are not necessarily equal.
	QVERIFY(n1 != n2);
}

void Tests::TestSerializeDeserialize()
{
	Network n1({2, 3, 2});
	Network n2({2, 3, 2});

	// Inital random state. (There is a 1/gazillion probability that the next line will fail.;))
	QVERIFY(n1 != n2);

	// Train network for one epoch in order to initialize biases, activations and derivatives
	CustomDataSet trainingSet("..\\XOR.training");
    n1.Train(trainingSet, 0.02, 0.9, 1, 0.02);

	// Serialize state from trained network and deserialize into another.
	n1.Serialize("XORTest.network");
	n2.DeSerialize("XORTest.network");

	// Deserialized network should now have the same state
	// as the serialized network
	QVERIFY(n1 == n2);
}

}
