import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

dir = os.getcwd()

path = os.path.join(dir,'data','asiacup.csv')

df = pd.read_csv(path)

df['ResultNo'] = df['Result'].apply(lambda x: 1 if x == 'Win' else 0) 

print(df.head())


#getting the required columns only
df_refine_X = df[['Team','Opponent','Ground','Year','Toss','Selection','Result']]
# getting the label out
y_label = df_refine_X.pop('Result')
#print(df_refine_X, y_label)


X_train, X_test, y_train, y_test = train_test_split(df_refine_X, y_label, test_size=0.2, random_state=42)

CATEGORICAL_COLUMNS = ['Team','Opponent','Ground','Year','Toss','Selection']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = X_train[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
