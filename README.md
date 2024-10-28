# Cloths-Price-Prediction
Predicting the price of cloths based on the material used,its state,brand and its category

Deploy a ML model to predict the price of cloths using various parameters like material,brand, state and its category.

The working of the project can be viewed from link given below:  http://127.0.0.1:5000

Dataset: It is almost a cleaned dataset with no missing values This dataset is taken From kaggle. This dataset is divided into training, validation and test dataset having 1000 rows for training and 500 each for validation and test dataset. The detailed description along with a link to download it is given here:https://www.kaggle.com/datasets/mrsimple07/clothes-price-prediction

ML model: This model is built using Randomforest Regressor model having 81% for training dataset and 81% accuracy for test dataset and around 70% accuracy for test dataset.

Flask: This model is deployed using Flask web application. Pycharm is used where the code for the model is inherited from jupyter notebook and collaborated in pycharm by using pickle module. Each of the Constructer is transfered from model.py to app.py using pkl extensions. To create files of pkl extensions the code is written in model.py.When you run the model.py in pycharm the pkl extension files for each constructor will be created automatically.
