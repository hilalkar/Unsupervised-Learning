import numpy as np
# import scipy.io
# from keras.utils import np_utils
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

veri=load_iris()
veri_input=veri["data"]
veri_target=veri["target"]
# veri = scipy.io.loadmat('iris_deneme1.mat')
train_inp,test_inp,train_tar,test_tar=(train_test_split
(veri_input,veri_target,test_size=0.3,shuffle=True,random_state=42))
# train_inp=veri['train_inp']
# train_tar=veri['train_tar']
# test_inp=veri['test_inp']
# test_tar=veri['test_tar']

# X=np.concatenate((train_inp,test_inp),axis=0)
# y=np.concatenate((train_tar,test_tar),axis=0)

kmeans1=KMeans(3)
kmeans1.fit(train_inp)
out_labels=kmeans1.labels_
print('Centers: ',kmeans1.cluster_centers_)
print('Predictions:')
print(kmeans1.predict([[50,30,15,2],[65,36,47,18]]))
tahmin=kmeans1.predict(test_inp)

plt.scatter(test_inp[:,1],test_tar,marker='o',c='blue')
plt.scatter(test_inp[:,1],tahmin,marker='.',c='red')
plt.legend(['Orjinal','Tahmin'])
plt.show()

acc = accuracy_score(test_tar, tahmin)
silhouette = silhouette_score(test_tar.reshape(-1,1), tahmin.reshape(-1,1))
print("Accuracy: %.2f%%" % (acc * 100))
print("Silhouette score: %.2f" % (silhouette))



# Bu kod, Iris veri kümesi üzerinde KMeans kümeleme algoritması kullanılarak yapılan
# bir sınıflandırma işlemini içeriyor. Ayrıca, test verisindeki tahminlerin doğruluğu
# ve siluet skoru hesaplanarak modelin performansı değerlendiriliyor