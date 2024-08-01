# Disaster_Response_Udacity_Project

# Project Description
This project is Udacity's Nanodegree program project introduced by Figure Eight. The dataset that was used includes prelablled texts and messages from real world disaster situations.
The aim of the project is to clean the datasets and build Natural Language Processing model to train on the dataset and use to classify messages based on real ones in crisis. 

Key steps of the project:
1) Build an ETL pipeline in order to process and clean the raw data in the datasets in order to feed it in the training and testing process in the next step by saving it in an SQLight (.db) file type.
2) Build an NLP pipeline - ML Algorithm to train it on classifying texts in various labels/categories.
3) Running a web app to test it on real world texts and show the results of the model classification.  


# Imported Libraries
1) The program uses Python 3 and above
2) Cleaning data: Numpy, Pandas
3) NLP processing: NLTK, Sciki-Learn
4) SQLite Database: SQLalchemy
5) Model Loading and Saving: Pickle
6) Web App and Data Visualization: Flask, Plotly


# How to execute the program:

1) To set up the database, train model and save the model, just follow and run the commands bellow for each part in the directory:
   - In order to process data and clean it, we use the ETL built pipeline: python data/process_data.py data/disaster_messages.csv
   data/disaster_categories.csv data/disaster_response_db.db
   - To train the model, use it for testing and save as a pickle file as a ML pipeline saved as .db file from the above step:
   python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl

2) After you run the above steps in the directory, you run this command in the app's directory to run your web app: python run.py


# Extra Materials and files:

I included two files in the models and data files that explain the ETL and NLP pipelines step by step to easily follow along:
  1- ETL Pipeline: to understand how ETL pipeline was built and implemented.
  2- NLP Pipeline: to understand how NLP pipeline was built and implemented using NLTK and Sciki-Learn
  
Feel free to test them yourself either to clean data or re-train the model and test the data using different classifiers or by changing the parameters for the Grid Search.

# Explanation of Files:

app/templates/: html files used to execute the web app

data/process_data.py: Extract Transform Load (ETL) pipeline used for data cleaning and storing data in a SQLite database.

models/train_classifier.py: A machine learning pipeline that loads data, etxracts features, trains a model, and saves the trained model as a .pkl file for later use.

run.py: This file can be used to launch the Flask web app used to classify disaster messages into categories.

# Acknowladgments:

Udacity for providing an this amazing and exciting Data Science Nanodegree Program
Figure Eight for providing the relevant dataset to train the model.

# Screenshots for showing some of the execution steps of the program:

1- This is where you can write a sentence in the box to test the performance of the model based on what type of the category it classifies the text by turning the category into a green light:
![Screenshot 6](https://github.com/user-attachments/assets/93735b62-a8af-415d-96e8-8c9864898405)


2- After you type it, click on classify message, there you can see the categories which the message belongs to as the color turns to green

![Screenshot 3](https://github.com/user-attachments/assets/e26c3a56-17a3-47c6-8535-59ba5b093139)

3- The main page shows some graphs about training dataset provided by Figure Eight.
 ![Screenshot 4](https://github.com/user-attachments/assets/969f8ac1-f428-4027-814b-183cc07137b2)
 
4- Sample run of process_data.py
![Screenshot 1 Udacity](https://github.com/user-attachments/assets/b9a0a6ca-40b4-4768-946a-eec9ae241ec8)


5- Sample run of train_classifier.py with precision, recall etc. for each category
![Screenshot 2](https://github.com/user-attachments/assets/19549f8a-3689-4c55-bcfc-b92dcd00c589)


6- Sample run of run.py
![Screenshot 5](https://github.com/user-attachments/assets/292c4f3a-90a7-436a-8acc-154c56b5eac0)


