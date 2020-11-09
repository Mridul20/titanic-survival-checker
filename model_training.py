import numpy as np
import pandas as pd
import pickle


from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier


dataset= pd.read_csv('D:\\Github\\Titanic-Machine-Learning-from-Disaster\\dataset\\updated.csv')

x = dataset.iloc[:,2:]
y = dataset.iloc[:,1]

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x,y)

pickle.dump(random_forest, open('D:\\Github\\Titanic-Machine-Learning-from-Disaster\\model.pkl','wb'))

model = pickle.load(open('D:\\Github\\Titanic-Machine-Learning-from-Disaster\\model.pkl','rb'))
print(model.predict(x),y)


print(x.head())
print(y.head())
z = [[3,0,1,1,22,1,1]]
print(random_forest.predict_proba(z))