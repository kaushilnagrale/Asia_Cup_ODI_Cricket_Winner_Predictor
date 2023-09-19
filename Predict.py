import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf


dir = os.getcwd()

path = os.path.join(dir,'data','asiacup.csv')

df = pd.read_csv(path)

df['ResultNo'] = df['Result'].apply(lambda x: 1 if x == 'Win' else 0) 

#print(df.head())



#getting the required columns only
df_refine_X = df[['Team','Opponent','Ground','Year','Toss','Selection','ResultNo']]
# getting the label out
y_label = df_refine_X.pop('ResultNo')
#print(df_refine_X, y_label)


X_train, X_test, y_train, y_test = train_test_split(df_refine_X, y_label, test_size=0.2, random_state=42)

CATEGORICAL_COLUMNS = ['Team','Opponent','Ground','Year','Toss','Selection']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = X_train[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(X_train, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(X_test, y_test, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
#print(result['accuracy'])  # the result variable is simply a dict of stats about our model


pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
#probs.plot(kind='hist', bins=20, title='predicted probabilities')





#Fetch from the frontend User sample for debugging
input_data = {
    'Team': 'Hong Kong',         # Replace with the actual values
    'Opponent': 'Australia',
    'Ground': 'Colombo(SSC)',
    'Year': 2022,
    'Toss': 'Australia',
    'Selection': 'Bowling'
}


# Function to make a single prediction
def predict_single_input(model, input_data):
    # Create a DataFrame with a single row containing the input data
    input_df = pd.DataFrame([input_data])
    
    # Use the input function to prepare the input for prediction
    single_input_fn = make_input_fn(input_df, pd.Series([0]), num_epochs=1, shuffle=False)
    
    # Use the trained linear estimator to predict the result for the single input
    single_pred_dicts = list(model.predict(single_input_fn))
    
    # Get the predicted probability for class 1 (Win)
    predicted_probability = single_pred_dicts[0]['probabilities'][1]
    
    # Convert the predicted probability to a class (1 if probability > 0.5, else 0)
    predicted_class = 1 if predicted_probability > 0.5 else 0
    
    return predicted_probability, predicted_class

# Make a prediction for the example input
predicted_probability, predicted_class = predict_single_input(linear_est, input_data)

print(f"Predicted Probability: {predicted_probability}")
print(f"Predicted Class: {'Win' if predicted_class == 1 else 'Lose'}")
