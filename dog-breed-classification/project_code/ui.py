# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import torch
import torch.nn as nn
import numpy as np
import time
import cv2
from os import path
import os
from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
import test_functions as tfc
import copy
import qimage2ndarray

model_path = '~/Desktop/DogBreedClassification/dog-breed-classification/result/model/'
img_folder = '~/Desktop/DogBreedClassification/test/'
img_path = None
test_img = None
transparenting = 50

heatmap0 = None
heatmap1 = None

with open("dogbreeds.txt", 'r') as f:
    dogbreeds = [line.rstrip('\n') for line in f]

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("*{\n"
            "font-family: century gothic;\n"
            "font-size:13px;\n"
            "background:white;\n"
            "}\n"
            "\n"
            "QPlainTextEdit{\n"
            "border:none;\n"
            "background:transparent;\n"
            "font-size:24px;\n"
            "font-weight:bold;\n"
            "color:#c0632f;\n"
            "font-variant: small-caps\n"
            "}\n"
            "\n"
            "QPushButton{\n"
            "background: white;\n"
            "font-size:14px;\n"
            "color:#c0632f;\n"
            "border: 1px solid #c0632f;\n"
            "border-radius:15px;\n"
            "}\n"
            "\n"
            "QPushButton:hover{\n"
            "color:white;\n"
            "border-radius:15px;\n"
            "border: 1px solid black;\n"
            "background:#d86f35;\n"
            "}\n"
            "\n"
            "QLabel{\n"
            "font-size:16px;\n"
            "background:white;\n"
            "color:#505050;\n"
            "}\n"
            "\n"
            "QLineEdit,QTextEdit{\n"
            "border:2px solid #c0632f;\n"
            "border-radius:10px;\n"
            "background:white;\n"
            "qproperty-alignment:AlignCenter;\n"
            "}\n"
            "\n"
            "#label_4{\n"
            "border:1px solid #505050;\n"
            "}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(192, 105, 121, 31))
        font = QtGui.QFont()
        font.setFamily("century gothic")
        font.setPointSize(-1)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(340, 105, 121, 31))
        font = QtGui.QFont()
        font.setFamily("century gothic")
        font.setPointSize(-1)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(510, 170, 251, 51))
        font = QtGui.QFont()
        font.setFamily("century gothic")
        font.setPointSize(-1)
        self.lineEdit.setFont(font)
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(520, 150, 159, 31))
        font = QtGui.QFont()
        font.setFamily("century gothic")
        font.setPointSize(-1)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(490, 105, 91, 31))
        font = QtGui.QFont()
        font.setFamily("century gothic")
        font.setPointSize(-1)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(510, 351, 251, 220))
        self.textEdit.setText("")
        self.textEdit.setObjectName("textEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(520, 330, 105, 31))
        self.label_2.setObjectName("label_2")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(270, 40, 301, 41))
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(520, 240, 151, 31))
        font = QtGui.QFont()
        font.setFamily("century gothic")
        font.setPointSize(-1)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(510, 260, 251, 51))
        font = QtGui.QFont()
        font.setFamily("century gothic")
        font.setPointSize(-1)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setText("")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(50, 175, 431, 361))
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.lineEdit.raise_()
        self.lineEdit_3.raise_()
        self.pushButton.raise_()
        self.pushButton_2.raise_()
        self.label.raise_()
        self.pushButton_3.raise_()
        self.textEdit.raise_()
        self.label_2.raise_()
        self.plainTextEdit.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.resnet_model = ''

        self.pushButton.clicked.connect(self.load_model)
        self.pushButton_2.clicked.connect(self.setImage)
        self.pushButton_3.clicked.connect(self.predict)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Load Model"))
        self.pushButton_2.setText(_translate("MainWindow", "Load Image"))
        self.label.setText(_translate("MainWindow", " Predict Dog Breed:"))
        self.pushButton_3.setText(_translate("MainWindow", "Predict"))
        self.label_2.setText(_translate("MainWindow", " Description:"))
        self.plainTextEdit.setPlainText(_translate("MainWindow", "Dog Breed Classification"))
        self.label_3.setText(_translate("MainWindow", " Confidence Level:"))

    def load_model(self):
        modelName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", model_path,
                                                            "Image Files (*.pth)")  # Ask for file
        print('fileName {}'.format(modelName))
        self.resnet_model = tfc.load_model(modelName)
        self.label_4.setText('{} loaded!'.format(modelName[-13:-4]))


    def setImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", img_folder,
                                                            "Image Files (*.png *.jpg *jpeg *.bmp *.tif)")  # Ask for file
        print('fileName {}'.format(fileName))
        global img_path, test_img
        img_path = fileName
        if fileName:  # If the user gives a file
            pixmap = QtGui.QPixmap(fileName)  # Setup pixmap with the provided image
            test_img = pixmap
            pixmap = pixmap.scaled(self.label_4.width(), self.label_4.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label_4.setPixmap(pixmap)  # Set the pixmap onto the label
            self.label_4.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center

    def predict(self):
        print('img_path {}'.format(img_path))
        tensor = tfc.tif_to_tensor(img_path)
        outputs = self.resnet_model(tensor)
        print('outputs {}'.format(outputs))
        scores = nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        print('scores {}'.format(scores))
        print('preds {}'.format(preds))
        preds_result = preds[0]
        preds_breed = dogbreeds[preds_result]
        scores = scores.data.cpu().numpy()
        confidence = scores[0][preds_result] * 100
        print('preds result {}'.format(preds_breed))

        if preds_result == 0:
            # print('This tissue is cancer negative with confidence of {:.2f}%!'.format(confidence))
            status = 'Negative'
        else:
            # print('This tissue is cancer positive with confidence of {:.2f}%!'.format(confidence))
            status = 'Positive'

        self.lineEdit.setText('{}'.format(preds_breed))
        self.lineEdit_3.setText('{:.2f}%'.format(confidence))

        with open("./breed_description/"+preds_breed+".txt", 'r') as f:
            breed_description = f.read()
        self.textEdit.setText('\n{}'.format(breed_description))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
