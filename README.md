# Data_science_project


1 - Data Collection

The quantity & quality of your data dictate how accurate our model is
The outcome of this step is generally a representation of data (Guo simplifies to specifying a table) which we will use for training
Using pre-collected data, by way of datasets from Kaggle, UCI, etc., still fits into this step
 
2 - Data Preparation

Wrangle data and prepare it for training
Clean that which may require it (remove duplicates, correct errors, deal with missing values, normalization, data type conversions, etc.)
Randomize data, which erases the effects of the particular order in which we collected and/or otherwise prepared our data
Visualize data to help detect relevant relationships between variables or class imbalances (bias alert!), or perform other exploratory analysis
Split into training and evaluation sets
 
3 - Choose a Model

Different algorithms are for different tasks; choose the right one
 
4 - Train the Model

The goal of training is to answer a question or make a prediction correctly as often as possible
Linear regression example: algorithm would need to learn values for m (or W) and b (x is input, y is output)
Each iteration of process is a training step
 
5 - Evaluate the Model

Uses some metric or combination of metrics to "measure" objective performance of model
Test the model against previously unseen data
This unseen data is meant to be somewhat representative of model performance in the real world, but still helps tune the model (as opposed to test data, which does not)
Good train/eval split? 80/20, 70/30, or similar, depending on domain, data availability, dataset particulars, etc.
 
6 - Parameter Tuning

This step refers to hyperparameter tuning, which is an "artform" as opposed to a science
Tune model parameters for improved performance
Simple model hyperparameters may include: number of training steps, learning rate, initialization values and distribution, etc.
 
7 - Make Predictions

Using further (test set) data which have, until this point, been withheld from the model (and for which class labels are known), are used to test the model; a better approximation of how the model will perform in the real world
