# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'testing.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(756, 504)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Utama = QtWidgets.QWidget(self.centralwidget)
        self.Utama.setStyleSheet("background-color:#DAD7CD;")
        self.Utama.setObjectName("Utama")
        self.Class = QtWidgets.QLabel(self.Utama)
        self.Class.setGeometry(QtCore.QRect(20, 400, 401, 41))
        self.Class.setText("")
        self.Class.setObjectName("Class")
        self.Camera = QtWidgets.QLabel(self.Utama)
        self.Camera.setGeometry(QtCore.QRect(20, 70, 331, 251))
        self.Camera.setStyleSheet("border: 5px solid #588157;\n"
"font-size: 12px;\n"
"border-radius: 10px;")
        self.Camera.setObjectName("Camera")
        self.Judul = QtWidgets.QLabel(self.Utama)
        self.Judul.setGeometry(QtCore.QRect(230, 10, 341, 61))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Judul.setFont(font)
        self.Judul.setObjectName("Judul")
        self.testing = QtWidgets.QPushButton(self.Utama)
        self.testing.setGeometry(QtCore.QRect(480, 350, 91, 31))
        self.testing.setStyleSheet("background-color:#344E41;\n"
"text-weight: bold;\n"
"color :white;\n"
"border-radius:10px;")
        self.testing.setObjectName("testing")
        self.Rmbg = QtWidgets.QLabel(self.Utama)
        self.Rmbg.setGeometry(QtCore.QRect(380, 70, 161, 101))
        self.Rmbg.setStyleSheet("border: 5px solid #588157;\n"
"font-size: 12px;\n"
"border-radius: 10px;")
        self.Rmbg.setObjectName("Rmbg")
        self.ambil_gambar = QtWidgets.QPushButton(self.Utama)
        self.ambil_gambar.setGeometry(QtCore.QRect(380, 350, 91, 31))
        self.ambil_gambar.setStyleSheet("background-color:#344E41;\n"
"text-weight: bold;\n"
"color :white;\n"
"border-radius:10px;")
        self.ambil_gambar.setObjectName("ambil_gambar")
        self.Result = QtWidgets.QLineEdit(self.Utama)
        self.Result.setGeometry(QtCore.QRect(20, 350, 331, 41))
        self.Result.setStyleSheet("border: 5px solid #588157;\n"
"font-size: 120%;\n"
"border-radius: 10px;\n"
"padding:5px 5px;\n"
"background: transparent;")
        self.Result.setObjectName("Result")
        self.grayscale = QtWidgets.QLabel(self.Utama)
        self.grayscale.setGeometry(QtCore.QRect(380, 210, 161, 101))
        self.grayscale.setStyleSheet("border: 5px solid #588157;\n"
"font-size: 12px;\n"
"border-radius: 10px;")
        self.grayscale.setObjectName("grayscale")
        self.LeftMenu_3 = QtWidgets.QWidget(self.Utama)
        self.LeftMenu_3.setGeometry(QtCore.QRect(590, 70, 131, 301))
        self.LeftMenu_3.setStyleSheet("background-color:#3A5A40;\n"
"border-radius:15px;")
        self.LeftMenu_3.setObjectName("LeftMenu_3")
        self.btn_Training = QtWidgets.QPushButton(self.LeftMenu_3)
        self.btn_Training.setGeometry(QtCore.QRect(20, 90, 91, 31))
        self.btn_Training.setStyleSheet("background-color:#A3B18A;\n"
"text-weight: bold;\n"
"border-radius:10px;")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../training.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_Training.setIcon(icon)
        self.btn_Training.setObjectName("btn_Training")
        self.btn_Testing = QtWidgets.QPushButton(self.LeftMenu_3)
        self.btn_Testing.setGeometry(QtCore.QRect(20, 160, 91, 31))
        self.btn_Testing.setStyleSheet("background-color:#A3B18A;\n"
"text-weight: bold;\n"
"border-radius:10px;")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../testing.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_Testing.setIcon(icon1)
        self.btn_Testing.setObjectName("btn_Testing")
        self.Menu = QtWidgets.QLabel(self.LeftMenu_3)
        self.Menu.setGeometry(QtCore.QRect(20, 10, 61, 61))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Menu.setFont(font)
        self.Menu.setStyleSheet("color: white;\n"
"")
        self.Menu.setObjectName("Menu")
        self.horizontalLayout.addWidget(self.Utama)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Camera.setText(_translate("MainWindow", "Camera"))
        self.Judul.setText(_translate("MainWindow", "SISTEM KLASIFIKASI SAMPAH"))
        self.testing.setText(_translate("MainWindow", "Testing"))
        self.Rmbg.setText(_translate("MainWindow", "Remove Bg"))
        self.ambil_gambar.setText(_translate("MainWindow", "Ambil Gambar"))
        self.Result.setPlaceholderText(_translate("MainWindow", "Hasil prediksi"))
        self.grayscale.setText(_translate("MainWindow", "Grayscale"))
        self.btn_Training.setText(_translate("MainWindow", "Training"))
        self.btn_Testing.setText(_translate("MainWindow", "Testing"))
        self.Menu.setText(_translate("MainWindow", "Menu"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())