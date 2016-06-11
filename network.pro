#-------------------------------------------------
#
# Project created by QtCreator 2015-12-22T09:50:06
#
#-------------------------------------------------

QT       += core testlib gui

TARGET = network
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

INCLUDEPATH += .\Regularization .\Activation


SOURCES += main.cpp \
	matrix.cpp \
	network.cpp \
	dataset.cpp \
	MNISTdataset.cpp \
	customdataset.cpp \
	tests.cpp \
	Regularization\L2.cpp\
	Activation\Sigmoid.cpp\
	Activation\ReLU.cpp\
	Activation\Tanh.cpp\
	Activation\Softmax.cpp\

HEADERS += \
	matrix.h \
	network.h \
	dataset.h \
	customdataset.h \
	MNISTdataset.h \
	autotest.h \
	tests.h \
	Regularization\IRegularization.h\
	Regularization\L2.h\
	Activation\Sigmoid.h\
	Activation\ReLU.h\
	Activation\Tanh.h\
	Activation\Softmax.h\

DISTFILES += \
	TODO

