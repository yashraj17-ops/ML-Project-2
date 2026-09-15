import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
df = pd.read_csv("Titanic-Dataset - Titanic-Dataset.csv")
print(type(df))
df.info()
df.head()
df.isna().sum()
df["Pclass"].value_counts()
df["Embarked"].value_counts()
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].ffill())
le = LabelEncoder()
le.fit(df["Gender"])
df["Gender"] = le.fit_transform(df["Gender"])
le.fit(df["Embarked"])
df["Embarked"] = le.fit_transform(df["Embarked"])
x = df[["Pclass", "Gender","SibSp", "Parch", "Embarked"]]
y = df["Survived"]
x.shape
y.shape
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
cr = classification_report(y_pred,y_test)
print(cr)
cm = confusion_matrix(y_pred,y_test)
print(cm)
acc = accuracy_score(y_pred,y_test)
print(acc)
