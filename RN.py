import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers.core import activation

class Redes:
    
    def __init__(self):
        
        self.df=pd.read_csv("diabetes.csv")
        self.x=self.df.iloc[:,0:8]
        self.y=self.df.iloc[:,8]
        self.model=0
        self.prediccion=0
        self.kfold=0
        self.accuracy=0
        
    def CrearModelo(self):
        
        self.model=Sequential()
        self.model.add(Dense(12,input_dim=8,activation='relu'))
        self.model.add(Dense(8,activation='relu'))
        self.model.add(Dense(1,activation='sigmoid'))
        
    def Compilar_Modelo(self):
        
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])    
        return self.model
    
    def Ejecutar_Modelo(self):
        
        self.model.fit(self.x,self.y,epochs=180,batch_size=16)
        
    def Evaluar_Modelo(self):
        
        _,self.accuracy=self.model.evaluate(self.x,self.y)
        print('Promedio {}'.format(self.accuracy*100))
        
    def predicciones(self):
           
        self.prediccion=self.model.predict(self.x)
        
        for i in range(20):
            
            print("Datos {} Esperado {}".format(np.round(self.prediccion[i]),self.y[i]))
        
        
        

modelo=Redes()
modelo.CrearModelo()
modelo.Compilar_Modelo()
modelo.Ejecutar_Modelo()
modelo.Evaluar_Modelo()  
modelo.predicciones()   
    
        
        
        
            
    

