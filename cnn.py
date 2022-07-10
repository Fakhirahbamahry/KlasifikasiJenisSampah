import sys
import program.preprocessing  as pp
import program.layer_cnn  as ac
import numpy as np
import pandas as pd
import copy
import os


#Generate the data
dataset = pp.ImageDataGenerator("Data")
X_train, y_train, X_test, y_test = dataset.load_dataset(rescale=255)
y_train_oneHot, y_test_oneHot = dataset.load_class_oneHot()

#Create a model
model = ac.Model([ac.Conv2d(padding=1,stride=2), ac.Relu(), ac.Maxpooling2D(ukuran_filter=2, stride=2), ac.Flatten(), ac.Dense([11]),])

#Training
model.summary()
model.fit(epochs=5000, X_input=X_train, y_input=y_train_oneHot, X_validation=X_test, y_validation=y_test_oneHot, callback=['plot_loss','plot_accuraacy'])

#Plot
model.plot()

#Test
#Labelling
classes = np.load('model/class.npy', allow_pickle=True)
Pred = []
for i in range(len(X_test)):
    model_baru = copy.deepcopy(model)
    predict = model_baru.pred(X_input=np.array([X_test[i]]))
    predict_arg_max = predict.argmax(axis=0)
    predict_class = classes[predict_arg_max[0]]
    prediksi = predict_class
    Pred.append((y_test[i],prediksi,))  
    
#File List    
labels = os.listdir(os.path.join("Data", "test"))
X = []
for label in labels:
    for file in os.listdir(os.path.join("Data", "test", label)):
        X.append((file))
        
#Create database
pdX = pd.DataFrame(X)
pdPred = pd.DataFrame(Pred)
result = pd.concat([pdX, pdPred], axis=1, join='inner')
result.columns = ['filename', 'label', 'prediction']
print(result)

result.to_csv('testing.csv')