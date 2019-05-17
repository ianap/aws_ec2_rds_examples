import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('titanic.csv')
include = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Survived']

# Create dummies and drop NaNs
df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)
df = df[include].dropna()

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp']]
y = df['Survived']

PREDICTOR = RandomForestClassifier(n_estimators=100).fit(X, y)

item_df = pd.DataFrame({'Pclass':[1], 'Sex':['1'], 'Age':[22], 'Fare':[80], 'SibSp':[2]})
score = PREDICTOR.predict_proba(item_df)

print('survival chances:' )
print(score[0,1])

print('death chances:')
print(score[0,0])
