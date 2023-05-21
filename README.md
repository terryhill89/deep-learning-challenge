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
### Optimization Attempts:     
#### 1. The First Model: 
I created by optimizing the model, adding another hidden layer to the model:
![Screenshot 2023-05-20 205301](https://github.com/terryhill89/deep-learning-challenge/assets/112741203/38aec731-528f-4fa7-bebf-03917ee00110)

The model was built with 3 hidden layers there was a relatively high number of input dimensions, The input layer was determined by (43 features) the shape of the training data. The number of neurons in the input layer is equal to 43. The activation chosen for the three hidden layers was Relu and for the output layer Sigmoid, which is suitable for binary classification tasks (whether funding was successful or not). After compling and training the model on the train data, I evaluated the model loss and accuracy measures on the test data. 

 ***Results:***
1st attempt at optimizing the model, add another hidden layer to the model.
![m](https://github.com/terryhill89/deep-learning-challenge/assets/112741203/11950adb-67cb-4989-8ca4-f02d147a6750)

#### 2. The Second Model: 
Decrease the number of layers, then decrease neurons per layer
![2](https://github.com/terryhill89/deep-learning-challenge/assets/112741203/4e991d0f-efdf-4ca7-a578-210b40f7a35d)

This model was built with 2 hidden layers, The input layer was determined by (43 features). The number of neurons in the input layer is equal to 43. The activation chosen for the two hidden layers was Relu and for the output layer Sigmoid, which is suitable for binary classification tasks (whether funding was successful or not). After compling and training the model on the train data, I evaluated the model loss and accuracy measures on the test data. 

   ***Results:*** 
   2nd optimization attempt failed. Decrease in Layers/neurons led to a slight decrease in the models accuracy.
 ![2](https://github.com/terryhill89/deep-learning-challenge/assets/112741203/afd22be8-d77d-4a9b-b5be-05b2b91b2e65)
 
#### 3. The Third Model: 
Start with the original model setup but drop some potentially unnecessary columns.
   - Drop STATUS and SPECIAL_CONSIDERATIONS Columns because they are simple Booleans describing a single factor they don't contribute that much to the decisions of the model.
![3](https://github.com/terryhill89/deep-learning-challenge/assets/112741203/744b586b-acfd-4b9b-97f2-2930118989ed)

This model was also built with 2 hidden layers, The input layer was determined by (40 features). The activation chosen for the two hidden layers was Relu and for the output layer Sigmoid, which is suitable for binary classification tasks (whether funding was successful or not). After compling and training the model on the train data, I evaluated the model loss and accuracy measures on the test data. 

***Results:*** 
3rd optimization attempt has failed. Removing the status and special considerations columns from the model had very little impact on the accuracy of the model only causing a bit of an increase in accuracy.
![3](https://github.com/terryhill89/deep-learning-challenge/assets/112741203/64b63152-0d72-4e12-9086-34ab04835df9)
The loss and accuracy were nearly the same as the first optimization model showing at L: 0.5712 and A: 0.7283 with a slight increase in accuracy showing at L: 0.5583 and A: 0.7296 for model 3.

#### 4. The Final Model: 
In the final attempt Compile, Train and Evaluate the model using Automated Tuning Approach to see if the accuracy will increase.
*!pip install keras_tuner and then
import the kerastuner library
import keras_tuner as kt*
![4](https://github.com/terryhill89/deep-learning-challenge/assets/112741203/3c0da8aa-5a01-410c-8b0d-192813343951)

Final optimization attempt was successful. There was a sufficent increase using the Automated Tuning Approach. I ran the automization model several times with no improvements. After adjusting the model a few times I seen high increase in accuracy. It did not reach 75% The highest score reached was 73%.

***Results:*** Automated Tuning Approach

![4](https://github.com/terryhill89/deep-learning-challenge/assets/112741203/eb0f7535-4d26-445c-ab58-b42635ed105e)

### Summary:
Our Goal was to develop a deep learning model to predict the success of applicants for funding by the nonprofit foundation *Alphabet Soup*. We performed data preprocessing, compiled, trained, evaluated and munipulated multiple models to achieve the best performance. On the whole, while the model never reached the target accuracy of 75%, it did come quite close at 73% for the best iteration. The Best model used for the final optimization was Automated Tuning Approach none of the other attempted models came close to the desired performance threshold.

