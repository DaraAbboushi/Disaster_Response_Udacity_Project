#importing libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """This function loads the input data and outputs a merged data:
    
    Input data consists of 2 files:
    messages_filepath is a path to the messages csv file
    categories_filepath is a path to the categories csv file
    
    Output data: is a merged the two files above stored in the 
    variable -df """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, on='id')
    
    return df


def clean_data(df):
    """This function cleans the merged data -df 
    to output a completely new cleaned merged data 
    Input data - df
    Output data - a clean df"""
    
    #clean the categories in order to make columns of them that have their own values 
    categories = pd.DataFrame(df['categories'])
    #we use split method to seperate the words from each other
    categories = categories['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # I used this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.split('-')[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    #iterate through it to extract the numbers by making the columns strings and then changing the type to int 
    for column in categories:
        categories[column] = pd.DataFrame(categories[column].astype(str).apply(lambda x: x.split('-')[1]))
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    #the old unclean categories column needs to be dropped and replaced with our new clean one
    df = df.drop(['categories'], axis=1)
    #we merge the two tables together 
    df = pd.concat([df, categories], axis=1)
    #the child_alone column has no binary values of 1, all numbers are 0 and so i deleted it
    df = df.drop(['child_alone'], axis=1)
    #Note the related column has to have a binary values of 0 or 1 however related column has three values so i replaced it with one
    df['related'] = df['related'].replace(2,1)
    
    # drop duplicates
    df = df.drop_duplicates()
        
    return df


def save_data(df, database_filename):
    """ This function saves our clean data using sqlite database and outputs a file that we can use 
    later and use it to feed the model 
    Input: df - cleaned data
    database_filename - database_filename is for sqlite database (.db for short) a type of file
    Output: saved clean data in a sqlite file type """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disaster_Response_Project', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'Disaster_Response_Project.db')


if __name__ == '__main__':
    main()
