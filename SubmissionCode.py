import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
a=np.genfromtxt('train_X.csv',delimiter=',')
b=np.genfromtxt('train_Y.csv',delimiter=',')
a=np.delete(a,0,0)
logreg = LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(a,b,test_size=0.3)
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
co=np.shape(x_test)[0]
y_pred.resize(co,1)
np.savetxt("predicted_test_Y.csv", y_pred, delimiter=",")

