#ifndef TESTS_H
#define TESTS_H

#include <QObject>
#include "autotest.h"

namespace UnitTest
{

class Tests : public QObject
{
	Q_OBJECT

public:
	Tests();
	~Tests();

private slots:
	void init();
	void cleanup();

	void TestNetworkCompareTopology();
	void TestNetworkCompareState();
	void TestSerializeDeserialize();
};

DECLARE_TEST(Tests)
}



#endif // TESTS_H
