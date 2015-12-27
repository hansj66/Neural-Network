#-------------------------------------------------
#
# Project created by QtCreator 2015-12-22T09:50:06
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = network
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    matrix.cpp \
    network.cpp \
    dataset.cpp \
    MNISTdataset.cpp \
    customdataset.cpp

HEADERS += \
    matrix.h \
    network.h \
    dataset.h \
    customdataset.h \
    MNISTdataset.h

DISTFILES += \
    TODO
