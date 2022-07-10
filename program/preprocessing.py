try:
    import cv2
    import numpy as np
    import os
except:
    print("Library tidak ditemukan !")
    print("Pastikan library cv2, os, numpy sudah terinstall")


class ImageDataGenerator:
    def __init__(self, lokasi):
        self.lokasi = lokasi

    def ambil_gambar(lokasi, jenis='', rescale=None):
        labels = os.listdir(os.path.join(lokasi, jenis))
        X = []
        y = []

        img_resize = 64
        for label in labels:
            for file in os.listdir(os.path.join(lokasi, jenis, label)):
                gambar = cv2.imread(os.path.join(lokasi, jenis, label, file))
                RGB_img_asli = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
                RGB_img = cv2.resize(RGB_img_asli, (img_resize, img_resize))
                R,c,d = RGB_img.shape
                r = np.zeros((R,c,1))
                g = np.zeros((R,c,1))
                b = np.zeros((R,c,1))
    
                r=RGB_img[:,:,0]
                g=RGB_img[:,:,1]
                b=RGB_img[:,:,2]
                result=((r/3)+(g/3)+(b/3))

                if rescale == None:
                    X.append(result)
                else:
                    X.append(result / rescale)
                y.append(label)

        return np.array(X), np.array(y)


    def load_dataset(self, rescale=1):
        x_train, y_train = ImageDataGenerator.ambil_gambar(lokasi=self.lokasi, jenis="train", rescale=rescale)
        x_test, y_test = ImageDataGenerator.ambil_gambar(lokasi=self.lokasi, jenis="test", rescale=rescale)
        self.y_train = y_train
        self.y_test = y_test
        np.save("model/class", self.load_class())

        return x_train, y_train, x_test, y_test

    def load_class(self):
        labels = os.listdir(os.path.join(self.lokasi, "train"))
        return labels

    def generate_oneHot(labels, kelas):
        arr_oneHot = []
        for k in kelas:
            oneHot = []
            for label in labels:
                if label == k:
                    oneHot.append(1)
                else:
                    oneHot.append(0)
            arr_oneHot.append(oneHot)
        return arr_oneHot
    def load_class_oneHot(self):
        labels = os.listdir(os.path.join(self.lokasi, "train"))
        y_train_oneHot = ImageDataGenerator.generate_oneHot(labels, self.y_train)
        y_test_onHot = ImageDataGenerator.generate_oneHot(labels, self.y_test)

        return y_train_oneHot, y_test_onHot