

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

# Load the saved model
loaded_model = tf.saved_model.load(saved_model_path)

# Make predictions using the loaded model (similar to your predict_single_input function)
predicted_probability, predicted_class = predict_single_input(loaded_model, input_data)

print(f"Predicted Probability: {predicted_probability}")
print(f"Predicted Class: {'Win' if predicted_class == 1 else 'Lose'}")
