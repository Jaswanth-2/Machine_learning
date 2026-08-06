from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    data=load_diabetes()
    x=pd.DataFrame(data.data,columns=data.feature_names)
    y=pd.Series(data.target,name='target')
    return x,y
