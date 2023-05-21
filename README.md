# Deep Learning Challenge
## Overview of the analysis:

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. 

The goal of the analysis is to find a model with your knowledge of machine learning and neural networks, that can predict whether applicants will be successful if funded by Alphabet Soup.

## Data Preprocessing
### Target Variables:
   - IS_SUCCESSFUL—Was the money used effectively
### Feature Variables:
   - EIN and NAME—Identification columns
   - APPLICATION_TYPE—Alphabet Soup application type
   - AFFILIATION—Affiliated sector of industry
   - CLASSIFICATION—Government organization classification
   - USE_CASE—Use case for funding
   - ORGANIZATION—Organization type
   - STATUS—Active status
   - INCOME_AMT—Income classification
   - SPECIAL_CONSIDERATIONS—Special consideration for application
   - ASK_AMT—Funding amount requested
### Variable(s) should be removed from the input data because they are neither targets nor features:
EIN and NAME are both identifications for the specific businesses that received funding in the past, they do not contribute directly to the success of the funding and therefor are neither targets nor features.
   - EIN and NAME—Identification columns 
   

## Compiling, Training, and Evaluating the Model
To get the data ready for the model I first explored the number of categories and each of the features it has, and the two features that had excessive number of categories were APPLICATION_TYPE and CLASSIFICATION. get_dummies() function to create a numerical binary-encoded representation of each of the categories for all categorical variables. Then I split the preprocessed data features and target arrays, and then split it to train and test sets. I then used the StandardScaler() function so the features can be used in the machine learning model.

    - Assessing outliers for ASK_AMT:
![e](https://github.com/terryhill89/deep-learning-challenge/assets/112741203/2afda97f-7426-4f91-b17d-d56dc012f9ae)
There are 8206 potential outliers out of 34299 records
That is 23.924895769555967% of the records
      ### Note: 
      Given that there are so many potential outliers I want to keep them and try different optimization methods.
      
   - The first model I created by optimizing the model, adding another hidden layer to the model:
![Screenshot 2023-05-20 205301](https://github.com/terryhill89/deep-learning-challenge/assets/112741203/38aec731-528f-4fa7-bebf-03917ee00110)
The model was built with 3 hidden layers there was a relatively high number of input dimensions, The input layer was determined by (43 features) the shape of the training data. The number of neurons in the input layer is equal to 43. The activation chosen for the three hidden layers was Relu and for the output layer Sigmoid, which is suitable for binary classification tasks (whether funding was successful or not). After compling and training the model on the train data, I evaluated the model loss and accuracy measures on the test data. 
Results:
![m](https://github.com/terryhill89/deep-learning-challenge/assets/112741203/11950adb-67cb-4989-8ca4-f02d147a6750)

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take in your attempts to increase model performance?
Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.






## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

   - EIN and NAME—Identification columns
   - APPLICATION_TYPE—Alphabet Soup application type
   - AFFILIATION—Affiliated sector of industry
   - CLASSIFICATION—Government organization classification
   - USE_CASE—Use case for funding
   - ORGANIZATION—Organization type
   - STATUS—Active status
   - INCOME_AMT—Income classification
   - SPECIAL_CONSIDERATIONS—Special considerations for application
   - ASK_AMT—Funding amount requested
   - IS_SUCCESSFUL—Was the money used effectively

## Instructions
### Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
   - What variable(s) are the target(s) for your model?
   - What variable(s) are the feature(s) for your model?
2. Drop the EIN and NAME columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

6. Use pd.get_dummies() to encode categorical variables.

7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Create the first hidden layer and choose an appropriate activation function.

4. If necessary, add a second hidden layer with an appropriate activation function.

5. Create an output layer with an appropriate activation function.

6. Check the structure of the model.

7. Compile and train the model.

8. Create a callback that saves the model's weights every five epochs.

9. Evaluate the model using the test data to determine the loss and accuracy.

10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

### Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

   - Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
     - Dropping more or fewer columns.
     - Creating more bins for rare occurrences in columns.
     - Increasing or decreasing the number of values for each bin.
     - Add more neurons to a hidden layer.
     - Add more hidden layers.
     - Use different activation functions for the hidden layers.
     - Add or reduce the number of epochs to the training regimen.
    
#### Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

### Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

1. Overview of the analysis: Explain the purpose of this analysis.

2. Results: Using bulleted lists and images to support your answers, address the following questions:

  - Data Preprocessing
    - What variable(s) are the target(s) for your model?
    - What variable(s) are the features for your model?
    - What variable(s) should be removed from the input data because they are neither targets nor features?

  - Compiling, Training, and Evaluating the Model
    - How many neurons, layers, and activation functions did you select for your neural network model, and why?
    - Were you able to achieve the target model performance?
    - What steps did you take in your attempts to increase model performance?

### Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
Step 5: Copy Files Into Your Repository
Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.

1. Download your Colab notebooks to your computer.

2. Move them into your Deep Learning Challenge directory in your local repository.

3. Push the added files to GitHub.
