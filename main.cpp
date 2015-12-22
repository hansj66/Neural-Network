#include "network.h"

#include <iostream>
#include <vector>
#include "trainingset.h"

using namespace std;

int main(int /* argc */ , char /* *argv[]*/ )
{
    // Example network has three layers with two input nodes, two nodes in the hidden layer
    // and one node in the output layer

    Network n({2, 2, 1});

    TrainingSet set("XOR.training");
    int iter = n.Train(set, 0.5, 0.001, 10000000);

    cout << "\nTraining result : ";
    if (iter == -1)
        cout << "FAILED" << endl;
    else
    {
        cout << "PASSED in "  << iter << " iterations." << endl;
    }


}
