import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.utils import np_utils
from keras.layers import Dense 
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import LabelEncoder 
import tensorflow

class verificacion:
    
    model=0
    
    def __init__(self):

        self.df= pd.read_csv("iris.csv")
        self.x=self.df.iloc[:,0:4].astype(float)
        self.y=self.df.iloc[:,4]

        self.encoder=LabelEncoder()
        self.encoder.fit(self.y)
        self.encoder_y=self.encoder.transform(self.y)
        print(self.encoder_y)
        self.dummy=np_utils.to_categorical(self.encoder_y)
        #self.model=0
        self.kfold=0
        self.estimador=0
        self.result=0
        
    def baseline_model(self):
        
        self.model=Sequential()
        self.model.add(Dense(8,input_dim=4,activation='relu'))
        self.model.add(Dense(3,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        return self.model
    
    def Calculo(self):
        
        
        self.estimator=KerasClassifier(build_fn=self.baseline_model,epochs=200,batch_size=5,verbose=0)
        self.kfold=KFold(n_splits=10,shuffle=True)
        print(self.dummy)
        self.result=cross_val_score(self.estimator,self.x,self.dummy,cv=self.kfold)
        print("Resultados {} y {}".format(self.result.mean()*100,self.result.std()*100))
        

v=verificacion()
v.baseline_model()
v.Calculo()
        