#include "network.h"

#include <iostream>
#include <vector>
#include "customdataset.h"
#include "MNISTdataset.h"
#include <QTime>

using namespace std;



void Example1_TrainXOR()
{
	CustomDataSet trainingSet("..\\XOR.training");

	forever
	{
		QTime time;
		time.start();

		Network n({2, 3, 2});

		int maxEpoch = 50000;
		int epoch = n.Train(trainingSet, 0.2, 50000);

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
			n.Run(trainingSet);
			n.ExportAsDigraph("d:\\network.gv");
			n.Serialize("d:\\XORWeightsAndBiases.network");
			break;
		}
	}
}


void Example2_TrainMNIST()
{
	MNISTDataSet trainingSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 1000);
	MNISTDataSet testSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 100);

	forever
	{
		QTime time;
		time.start();

		Network n({784, 63, 10});

		int maxEpoch = 1000;
		int epoch = n.Train(trainingSet, 0.2, maxEpoch);

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
			n.Run(testSet);
			n.ExportAsDigraph("d:\\network.gv");
			n.Serialize("d:\\MNISTWeightsAndBiases.network");
			break;
		}
	}
}

int main(int /* argc */ , char /* *argv[]*/ )
{
	Example2_TrainMNIST();
}
