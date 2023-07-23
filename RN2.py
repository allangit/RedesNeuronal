import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers.core import activation


class Clasificacion:
    
    
    def __init__(self):
        
        self.df
        self.x
        self.y
        self.model
        self.prediccion