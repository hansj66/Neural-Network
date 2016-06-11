#include "MNISTdataset.h"
#include <iostream>
#include <fstream>
#include <QDir>

using namespace std;

quint32 MNISTDataSet::Parameter(unsigned char * memory)
{
	quint32 p = 0;
	p |= *(memory);
	p = p << 8;
	p |= *(memory+1);
	p = p << 8;
	p |= *(memory+2);
	p = p << 8;
	p |= *(memory+3);
	return p;
}

MNISTDataSet::MNISTDataSet(string input, string output, quint32 maxImages)
{
	
	ifstream trainingSet(input,  ios::in|ios::binary|ios::ate);

	if (!trainingSet.is_open())
	{
		throw string("Error. Unable to open training set file");
	}

	/* File format:
		[offset] [type]          [value]          [description]
		0000     32 bit integer  0x00000803(2051) magic number
		0004     32 bit integer  60000            number of images
		0008     32 bit integer  28               number of rows
		0012     32 bit integer  28               number of columns
		0016     unsigned byte   ??               pixel
		0017     unsigned byte   ??               pixel
		........
		xxxx     unsigned byte   ??               pixel
	*/
	streampos size = trainingSet.tellg();
	char * memblockSet = new char [size];
	trainingSet.seekg (0, ios::beg);
	trainingSet.read (memblockSet, size);

	quint32 magicNumber = Parameter((unsigned char*)memblockSet);
	if (2051 != magicNumber)
	{
		cout << "Error. " << input << " is not a MNIST training set." << endl;
		trainingSet.close();
		return;
	}
	quint32 nImages = Parameter((unsigned char*)memblockSet+4);
	if (-1 != maxImages)
		nImages = maxImages;
	quint32 nRows = Parameter((unsigned char*)memblockSet+8);
	quint32 nColumns = Parameter((unsigned char*)memblockSet+12);

	cout << "MNIST file (" << input << ") read " << nImages << " images. Size: " << nColumns << "x" << nRows << endl;

	cout << "Reading training images... ";
	_set.resize(nImages);
	qint32 offset = 16;
	for (quint32 i=0; i<nImages; i++)
	{
		for (quint32 b=0; b<nRows*nColumns; b++)
		{
			Input(i).push_back(*(memblockSet+offset));
			offset++;
		}
	}
	cout << "Done." << endl;

	ifstream trainingLabels(output,  ios::in|ios::binary|ios::ate);
	if (!trainingLabels.is_open())
	{
		cout << "Unable to open training labels file: " << output << endl;
		return;
	}

	/* File format:
		[offset] [type]          [value]          [description]
		0000     32 bit integer  0x00000801(2049) magic number (MSB first)
		0004     32 bit integer  60000            number of items
		0008     unsigned byte   ??               label
		0009     unsigned byte   ??               label
		........
		xxxx     unsigned byte   ??               label
		The labels values are 0 to 9.
	*/

	size = trainingLabels.tellg();
	char * memblockLabels = new char [size];
	trainingLabels.seekg (0, ios::beg);
	trainingLabels.read (memblockLabels, size);

	magicNumber = Parameter((unsigned char*)memblockLabels);
	if (2049 != magicNumber)
	{
		cout << "Error. " << input << " is not a MNIST training label file." << endl;
		trainingLabels.close();
		return;
	}

	quint32 nLabels = Parameter((unsigned char*)(memblockLabels+4));
	if (-1 != maxImages)
		nLabels = maxImages;

	cout << "MNIST file (" << output << ") read " << nLabels << " labels"<< endl;

	vector<double> zero =   {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	vector<double> one =    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
	vector<double> two =    {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
	vector<double> three =  {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
	vector<double> four =   {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
	vector<double> five =   {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
	vector<double> six =    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
	vector<double> seven =  {0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
	vector<double> eight =  {0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
	vector<double> nine =   {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

	cout << "Reading training labels... ";
	offset = 8;
	for (quint32 i=0; i<nLabels; i++)
	{
		unsigned char label = (unsigned char)*(memblockLabels+offset);
		switch(label)
		{
			case 0: Output(i) = zero; break;
			case 1: Output(i) = one; break;
			case 2: Output(i) = two; break;
			case 3: Output(i) = three; break;
			case 4: Output(i) = four; break;
			case 5: Output(i) = five; break;
			case 6: Output(i) = six; break;
			case 7: Output(i) = seven; break;
			case 8: Output(i) = eight; break;
			case 9: Output(i) = nine; break;
			default: std::cout << "Corrupt file. Aborting.\n";
					 exit(1);
		}
		offset++;
	}
	cout << "Done." << endl;

	trainingSet.close();
	trainingLabels.close();
	delete[] memblockSet;
	delete[] memblockLabels;

}


MNISTDataSet::~MNISTDataSet()
{

}
