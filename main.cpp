#include "network.h"

#include <QCoreApplication>
#include <iostream>
#include <vector>
#include "customdataset.h"
#include "MNISTdataset.h"
#include <QTime>
#include <QFile>
#include <ctime>
#include <cstdlib>
#include <QTextStream>
#include <QTimer>
#include "autotest.h"

using namespace std;



void Example1_TrainCustom()
{
	CustomDataSet trainingSet("logic.train");

	forever
	{
		QTime time;
		time.start();

		Network n({2, 5, 5, 2});

		int maxEpoch = 50000;
		int epoch = n.Train(trainingSet, 0.01, 0.7, 50000, 0.02);

		if (epoch == maxEpoch)
		{
			cout << "\nGiving up. Restarting..." << endl;
			continue;
		}
		else
		{
			cout << "\nTraining result : ";
			cout << "Finished after "  << epoch << " epochs (" << time.elapsed() << " ms)" << endl;
			cout << "Verifying against test set...\n";
			n.Run(trainingSet, "Training set", true);
			n.ExportAsDigraph("d:\\network.gv");
			n.Serialize("d:\\XORWeightsAndBiases.network");
			break;
		}
	}
}


void Example2_TrainMNIST(MNISTDataSet & trainingSet, MNISTDataSet & testSet)
{
	QTime time;
	time.start();

	Network n({784, 128, 10});
	if (QFile("MNISTWeightsAndBiases.network").exists())
		n.DeSerialize("MNISTWeightsAndBiases.network");

	int batchSize = 3000;
	DataSet batch = trainingSet.CreateBatch(batchSize);
	int maxEpoch = 100;
	try
	{
		n.Train(batch, 0.02 , 0.7, maxEpoch, 0.2);
		n.Serialize("MNISTWeightsAndBiases.network");

		int trainingFit = n.Run(batch, "Training set");
		int testFit = n.Run(testSet, "Test set");

		QString logFileName = "network.training.log";
		QFile file(logFileName);
		if ( !file.open(QIODevice::Append) )
		{
			cout << "Error opening '" << logFileName.toStdString() << "' for writing." << endl;
			exit(1);
		}

		QTextStream stream( &file );
		stream << trainingFit << "/" << batch.Size() << "," << testFit << "/" << testSet.Size() << endl;
		file.close();

		n.ExportAsDigraph("network.gv");


	}
	catch(string ex)
	{
		cout << ex << " Skipping batch..." << endl;
	}
}

int main(int argc, char *argv[])
{
	Q_UNUSED(argc);
	Q_UNUSED(argv);
	try
	{
		srand(time(0));

//		Example1_TrainCustom();


#ifdef RUN_UNIT_TESTS
		return UnitTest::run(1 /* ignore arguments */, argv);
#endif

		MNISTDataSet trainingSet(".\\TrainingSets\\MNITS\\train-images.idx3-ubyte", ".\\TrainingSets\\MNITS\\train-labels.idx1-ubyte", 59992);
		MNISTDataSet testSet(".\\TrainingSets\\MNITS\\t10k-images.idx3-ubyte", ".\\TrainingSets\\MNITS\\t10k-labels.idx1-ubyte", 9900);
		forever
		{
			Example2_TrainMNIST(trainingSet, testSet);
		}
	}
	catch (string ex)
	{
		cout << ex << endl;
	}
}
