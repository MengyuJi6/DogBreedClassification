# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'kaggle_ui.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
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
        MainWindow.resize(640, 450)
        key.pngMainWindow.setStyleSheet("#MainWindow { background-color: yellow; }")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(45, 80, 99, 27))
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(45, 140, 99, 27))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(45, 200, 99, 27))
        self.pushButton_3.setObjectName("pushButton_3")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(200, 20, 400, 300))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(340, 330, 300, 111))
        self.label_2.setObjectName("label_2")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 641, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.resnet_model = ''

        self.pushButton.clicked.connect(self.load_model)
        self.pushButton_2.clicked.connect(self.setImage)
        self.pushButton_3.clicked.connect(self.predict)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DogBreedClassification"))
        self.pushButton.setText(_translate("MainWindow", "Load Model"))
        self.pushButton_2.setText(_translate("MainWindow", "Load Image"))
        self.pushButton_3.setText(_translate("MainWindow", "Predict"))
        self.label_2.setText(_translate("MainWindow", ""))

    def load_model(self):
        modelName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", model_path,
                                                            "Image Files (*.pth)")  # Ask for file
        print('fileName {}'.format(modelName))
        self.resnet_model = tfc.load_model(modelName)
        self.label_2.setText('{} loaded!'.format(modelName[-13:-4]))


    def setImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", img_folder,
                                                            "Image Files (*.png *.jpg *jpeg *.bmp *.tif)")  # Ask for file
        print('fileName {}'.format(fileName))
        global img_path, test_img
        img_path = fileName
        if fileName:  # If the user gives a file
            pixmap = QtGui.QPixmap(fileName)  # Setup pixmap with the provided image
            test_img = pixmap
            pixmap = pixmap.scaled(self.label.width(), self.label.height(),
                                   QtCore.Qt.KeepAspectRatio)  # Scale pixmap
            self.label.setPixmap(pixmap)  # Set the pixmap onto the label
            self.label.setAlignment(QtCore.Qt.AlignCenter)  # Align the label to center
        self.label_2.setText('Test image loaded!')

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

        self.label_2.setText('Dog breed: {}\nConfidence Level: {:.2f}%'.format(preds_breed, confidence))

def hello():
    print('hello world {}'.format(img_path))

def qt_image_to_array(img, share_memory=False):
    """ Creates a numpy array from a QImage.

        If share_memory is True, the numpy array and the QImage is shared.
        Be careful: make sure the numpy array is destroyed before the image,
        otherwise the array will point to unreserved memory!!
    """
    assert isinstance(img, QtGui.QImage), "img must be a QtGui.QImage object"
    assert img.format() == QtGui.QImage.Format.Format_RGB32, \
        "img format must be QImage.Format.Format_RGB32, got: {}".format(img.format())

    img_size = img.size()
    buffer = img.constBits()

    # Sanity check
    n_bits_buffer = len(buffer) * 8
    n_bits_image  = img_size.width() * img_size.height() * img.depth()
    assert n_bits_buffer == n_bits_image, \
        "size mismatch: {} != {}".format(n_bits_buffer, n_bits_image)

    assert img.depth() == 32, "unexpected image depth: {}".format(img.depth())

    # Note the different width height parameter order!
    arr = np.ndarray(shape  = (img_size.height(), img_size.width(), img.depth()//8),
                     buffer = buffer,
                     dtype  = np.uint8)

    if share_memory:
        return arr
    else:
        return copy.deepcopy(arr)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
