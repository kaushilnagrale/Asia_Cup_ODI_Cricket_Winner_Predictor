import tensorflow as tf
import os 
import pandas as pd
import numpy as np


dir = os.getcwd()
path = os.path.join(dir, 'saved_model', '1695128323')

# Load the saved model
loaded_model = tf.saved_model.load(path)

inference_fn = loaded_model.signatures["serving_default"]

# Now you can use inference_fn to make predictions
# Create tensors for input data based on the model's input signature
team = tf.constant(["Hong Kong"])
opponent = tf.constant(["Australia"])
ground = tf.constant(["Colombo(SSC)"])
year = tf.constant([2022], dtype=tf.int64)
toss = tf.constant(["Australia"])
selection = tf.constant(["Bowling"])

# Create a dictionary with the "inputs" key
input_data = {
    "Team": team,
    "Opponent": opponent,
    "Ground": ground,
    "Year": year,
    "Toss": toss,
    "Selection": selection
}

# Call the inference function with the input_data dictionary
output = inference_fn(**input_data)

# Extract the prediction
predictions = output["output_name"].numpy()  # Replace "output_name" with the actual output name
print(predictions)








def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

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


# Make predictions using the loaded model (similar to your predict_single_input function)
#predicted_probability, predicted_class = predict_single_input(loaded_model, input_data)

#print(f"Predicted Probability: {predicted_probability}")
#print(f"Predicted Class: {'Win' if predicted_class == 1 else 'Lose'}")
