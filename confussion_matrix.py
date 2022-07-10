from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

hasil = pd.read_csv("testing.csv")
l = hasil["label"]
p = hasil["prediction"]

#analisa dengan confussion matrix
def display_conf(l,p):
    sns.heatmap(confusion_matrix(l,p),annot=True,linewidths=3,cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Prediction')
    plt.show()

# Memanggil fungsi untuk menampilkan visualisasi confusion matrix
display_conf(l,p)

# print(f'R2 Score : {r2_score(l,p)}')
# print('Classification Report :')
# print(classification_report(l,p))
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(l, p)))
# print('Micro Precision: {:.2f}'.format(precision_score(l, p, average='micro')))
# print('Micro Recall: {:.2f}'.format(recall_score(l, p, average='micro')))
# print('Micro F1-score: {:.2f}\n'.format(f1_score(l, p, average='micro')))

# print('Macro Precision: {:.2f}'.format(precision_score(l, p, average='macro')))
# print('Macro Recall: {:.2f}'.format(recall_score(l, p, average='macro')))
# print('Macro F1-score: {:.2f}\n'.format(f1_score(l, p, average='macro')))

# print('Weighted Precision: {:.2f}'.format(precision_score(l, p, average='weighted')))
# print('Weighted Recall: {:.2f}'.format(recall_score(l, p, average='weighted')))
# print('Weighted F1-score: {:.2f}'.format(f1_score(l, p, average='weighted')))

print('\nClassification Report\n')
print(classification_report(l, p, target_names=['Kaca', 'Kertas atau Kardus', 'Logam','Plastik','Plastik Lembaran']))