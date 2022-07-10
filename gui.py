from platform import release
import sys
import numpy as np
from PyQt5 import  uic,QtWidgets,QtCore,QtGui
from PyQt5.QtWidgets import QMessageBox
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import os
import time
import copy
import shutil
from rembg.bg import remove
from PIL import Image
from program.layer_cnn import Model, Conv2d, Maxpooling2D, Flatten, Dense, Relu
import program.layer_cnn as lc
from program.preprocessing import ImageDataGenerator
from program.callback import ProgressBar
from datetime import datetime



class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('Design/training.ui', self)
        self.show()
        self.setWindowTitle('Sistem Klasifikasi Sampah')
        self.model = Model([
            Conv2d(padding=1, stride=2),
            Relu(),
            Maxpooling2D(ukuran_filter=2, stride=2),
            Flatten(),
            Dense([11]),
        ])
        self.epochs = 5000
        self.btn_training.clicked.connect(self.training)
        self.btn_testing.clicked.connect(self.window2)
        self.btn_dataset.clicked.connect(self.window3)

    def training(self):
        self.tombol_train.clicked.connect(self.proses_training)

        dir = 'Data/train/'
        list = os.listdir(dir)
        teks = ''
        for i in list:
            jumlah = str(len(os.listdir(dir+i)))
            teks = teks + "Class {} Memiliki {} Data".format(i, jumlah) + '\n'

        self.textEdit.append(teks)

    def proses_training(self):
        QMessageBox.warning(self, "Informasi", "Memulai Proses Training. Mohon tunggu sebentar..")
        self.progres.setText("Memulai Proses Konvolusi...")
        dataset = ImageDataGenerator("Data")
        X_train, y_train, X_test, y_test = dataset.load_dataset(rescale=255)
        y_train_oneHot, y_test_oneHot = dataset.load_class_oneHot()

        self.progres.setText("Memulai Proses Training. Mohon tunggu sebentar..")
        model = self.model
        teks = model.summary(input_shape=(64, 64))

        model.fit(epochs=self.epochs, X_input=X_train, y_input=y_train_oneHot, X_validation=X_test, y_validation=y_test_oneHot,
                  callback=[''])
        model.plot()
        self.textEdit.append(teks)
        np.save("model/class", dataset.load_class())

        self.progres.setText("Proses Training Selesai.")
        self.Akurasi.setText("Akurasi yang dihasilkan "+ str(model.acc[len(model.acc)-1]))

    def window2(self):                                             
        self.w = MyWindow2()
        self.w.show()
        self.hide()

    def window3(self):                                             
        self.w3 = MyWindow3()
        self.w3.show()
        self.hide()

class MyWindow2(QtWidgets.QMainWindow):                           
    def __init__(self):
        super().__init__()
        uic.loadUi('Design/testing.ui', self)
        self.show()
        self.setWindowTitle('Sistem Klasifikasi Sampah')
        self.btn_Training.clicked.connect(self.window)
        self.btn_Testing.clicked.connect(self.window2)
        self.btn_Dataset.clicked.connect(self.window3)

        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Webcam')

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()
        self.ambil_gambar.clicked.connect(self.ambilgambar)
        self.testing.clicked.connect(self.Testing)
        



    @pyqtSlot(np.ndarray)
    def update_image(self, frame):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(frame)
        self.gambar_frame=frame
        self.Camera.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


    def ambilgambar(self,cap, rescale=None):                                            
        # self.thread.cap.release()
        self.Class.setText("")
        now = datetime.now()
        datestring = now.strftime("%Y%m%d_%H%M%S")
        output_path = "Percobaan/IMG_"+datestring+"_bg.png"
        gb = "Percobaan/IMG_"+datestring+"_gray.png"
        var = cv2.imwrite("Percobaan/IMG_"+datestring+"_asli.jpg",self.gambar_frame)
        im = Image.fromarray(self.gambar_frame)
        output = remove(im)
        output.save(output_path)
        self.Rmbg.setPixmap(QPixmap(output_path))
        imgg = cv2.imread(output_path)
        asli = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
        R,c,d = asli.shape
        r = np.zeros((R,c,1))
        g = np.zeros((R,c,1))
        b = np.zeros((R,c,1))

        r=asli[:,:,0]
        g=asli[:,:,1]
        b=asli[:,:,2]
        result=((r/3)+(g/3)+(b/3))

        if rescale == None:
            gray = result
        else:
            gray = result / rescale
        cv2.imwrite(gb,gray)
        self.grayscale.setPixmap(QPixmap(gb))

        return gray

    def Testing(self,gray):
        img_resize = 64

        gray_resize = cv2.resize(gray, (img_resize, img_resize))
        # print(np.array(np.array()).shape)
        train = MyWindow()
        classes = np.load('model/class.npy', allow_pickle=True)
        Pred = []
        model_baru = copy.deepcopy(train.model)
        predict = model_baru.pred(X_input=np.array(np.array([gray_resize])))
        print(predict)
        predict_arg_max = predict.argmax(axis=0)
        predict_class = classes[predict_arg_max[0]]
        print(predict[predict_arg_max[0]], predict_class)
        prediksi = predict_class
        self.Result.setText(prediksi)
        self.Class.setText('Benda pada gambar terdeteksi kategori '+str(prediksi))
        # self.Class.setText("Benda pada gambar terdeteksi kategori {} dengan nilai {}%".format(prediksi, round(predict[predict_arg_max[0]][0] * 100)))
        return prediksi

    def window(self):                                             
        self.w2 = MyWindow()
        self.w2.show()
        self.hide()

    def window2(self):                                             
        self.w = MyWindow2()
        self.w.show()
        self.hide()
    
    def window3(self):                                             
        self.w3 = MyWindow3()
        self.w3.show()
        self.hide()

class MyWindow3(QtWidgets.QMainWindow):                           
    def __init__(self):
        super().__init__()
        uic.loadUi('Design/dataset.ui', self)
        self.show()
        self.setWindowTitle('Sistem Klasifikasi Sampah')
        self.btn_Training.clicked.connect(self.window)
        self.btn_Testing.clicked.connect(self.window2)
        self.btn_Dataset.clicked.connect(self.window3)

        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Webcam')

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()
        self.ambil_gambar.clicked.connect(self.ambilgambar)
        self.create_data.clicked.connect(self.simpangambar)
        
        



    @pyqtSlot(np.ndarray)
    def update_image(self, frame):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(frame)
        self.gambar_frame=frame
        self.Camera.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def ambilgambar(self,cap, rescale=None):                                            
        # self.thread.cap.release()
        self.Keterangan.setText("")
        now = datetime.now()
        datestring = now.strftime("%Y%m%d_%H%M%S")
        output_path = "Percobaan/IMG_c.png"
        var = cv2.imwrite("Percobaan/IMG_"+datestring+"_dataasli.jpg",self.gambar_frame)
        im = Image.fromarray(self.gambar_frame)
        output = remove(im)
        output.save(output_path)
        self.Rmbg.setPixmap(QPixmap(output_path))
    
    def simpangambar(self): 
        input = self.Kategori.text()
        now = datetime.now()
        datestring = now.strftime("%Y%m%d_%H%M%S")
        if input == 'kaca'or input =='Kaca'or input =='KACA' :
            original = r'Percobaan/IMG_c.png'
            target = r'Data/train/Kaca/IMG_'+datestring+'.png'
            shutil.move(original, target)
            self.Keterangan.setText("Berhasil di simpan")
        elif input == 'kertas atau kardus'or input =='Kertas atau Kardus'or input =='Kertas Atau Kardus'or input =='KERTAS ATAU KARDUS':
            original = r'Percobaan/IMG_c.png'
            target = r'Data/train/Kertas atau Kardus/IMG_'+datestring+'.png'
            shutil.move(original, target) 
            self.Keterangan.setText("Berhasil di simpan")
        elif input == 'logam'or input =='Logam'or input =='LOGAM':
            original = r'Percobaan/IMG_c.png'
            target = r'Data/train/Logam/IMG_'+datestring+'.png'
            shutil.move(original, target) 
            self.Keterangan.setText("Berhasil di simpan")
        elif input == 'plastik'or input =='Plastik'or input =='PLASTIK':
            original = r'Percobaan/IMG_c.png'
            target = r'Data/train/Plastik/IMG_'+datestring+'.png'
            shutil.move(original, target) 
            self.Keterangan.setText("Berhasil di simpan")
        elif input == 'plastik lembaran'or input =='Plastik Lembaran'or input =='PLASTIK LEMBARAN'or input =='Plastik lembaran':
            original = r'Percobaan/IMG_c.png'
            target = r'Data/train/Plastik Lembaran/IMG_'+datestring+'.png'
            shutil.move(original, target) 
            self.Keterangan.setText("Berhasil di simpan")
        else:
            self.Keterangan.setText("Maaf kategori tidak ditemukan")
        
    def window(self):                                             
        self.w2 = MyWindow()
        self.w2.show()
        self.hide()

    def window2(self):                                             
        self.w = MyWindow2()
        self.w.show()
        self.hide()
    
    def window3(self):                                             
        self.w3 = MyWindow3()
        self.w3.show()
        self.hide()

class VideoThread(QThread):
    
    change_pixmap_signal = pyqtSignal(np.ndarray)
    cap = cv2.VideoCapture(0)

    def run(self):
        # capture from web cam
        while True:
            ret, cv_img = self.cap.read()
            if ret:
                frame1 = cv2.resize(cv_img,(250,200))
                frame = cv2.flip(frame1, 1)
                self.change_pixmap_signal.emit(frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                # showPic = cv2.imwrite("Percobaan/5.jpg",frame)  
    
         
    

    
    # def removeBG(self):
    #     input_path = '4.jpg'
    #     output_path = '1.png'
    #     input = Image.open(input_path)
    #     output = remove(input)
    #     output.save(output_path)


app = QtWidgets.QApplication(sys.argv)
w = MyWindow()
w.show()
sys.exit(app.exec_())
