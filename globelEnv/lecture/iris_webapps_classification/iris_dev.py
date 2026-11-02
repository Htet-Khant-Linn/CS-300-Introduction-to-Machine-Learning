from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

X,y=load_iris(return_X_y=True,as_frame=True)

#df=pd.read_csv('dataset/Iirs.csv')

#lbl=LabelEncoder()
#df['Species']=lbl.fit_transform([df['Species']])
#X=df.drop(columns=['Id','Species'],axis=1)
#y=df['Species']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)
lr=LogisticRegression()
lr.fit(X_train,y_train)
pred_value=lr.predict(X_test)




import pickle #joblib #h5
with open('iris_model.pkl','wb') as f:
    pickle.dump(lr,f)

print("Accuracy score ",accuracy_score(pred_value,y_test)*100)
print("Saving model is done.")
