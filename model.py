import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import joblib
import pickle

dataset = pd.read_csv('csv_result-Autism-Child-Data.csv')
dataset=dataset.drop(['id','age_desc','contry_of_res','used_app_before','result'],axis=1)
dataset = dataset.replace('?', np.nan)
dataset.isnull().sum()
dataset['relation'].unique()
dataset['age'] = pd.to_numeric(dataset['age'], errors='coerce')
avg = round((dataset['age'].mean()))
avg = float(avg)
dataset['age'].fillna(avg, inplace=True)
dataset['ethnicity'].fillna("Others", inplace=True)
dataset['relation'].fillna("Not Mentioned", inplace=True)

le1 = LabelEncoder()
dataset['gender'] = le1.fit_transform(dataset['gender'])
dataset['jundice'] = le1.fit_transform(dataset['jundice'])
dataset['ethnicity'] = le1.fit_transform(dataset['ethnicity'])
dataset['austim'] = le1.fit_transform(dataset['austim'])
dataset['relation'] = le1.fit_transform(dataset['relation'])
dataset['Class/ASD'] = le1.fit_transform(dataset['Class/ASD'])

dataset.reset_index(drop=True,inplace=True)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


X, y = make_classification(n_samples=100, random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
joblib.dump(clf, 'model.pkl')

cols=["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score","age","gender","ethnicity","jaundice","autism","relation"]
joblib.dump(cols, 'model_cols.pkl')