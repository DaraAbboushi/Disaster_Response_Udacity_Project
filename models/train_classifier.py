import sys
import nltk
import pickle
nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, f1_score
import re 
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    """ The load function takes the saved cleaned sqlite database and outputs/read into a merged file
    of messages and categories
    Input: database_filepath
    Output: 
    X - messages - input varaible
    y - categories of the messages, what category the message falls in - output variable
    category_names - category column names of y """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Disaster_Response_Project', con=engine)
    
    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns.values
    return X, y, category_names


class VerbAndNounExtractor(BaseEstimator, TransformerMixin):
    """     Verb And Noun Extractor
    
    This class extract the starting Verb And Noun of a list of words,
    creating a new feature for the ML classifier """
    
    def noun_verb_extractor(self, text):
        words = word_tokenize(text)
        for word in words:
            pos_tags = nltk.pos_tag(tokenize(word))
            chosen_word, word_tag = pos_tags[0]        
            if word_tag in ['VB','VBP', 'NN'] and chosen_word not in stopwords.words('english') and chosen_word not in ["@", "''", "*", "[", "%", "'m", "'re", "'m'", "."]:
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.noun_verb_extractor)
        return pd.DataFrame(X_tagged)
 


def tokenize(text):
    """ This function cleans the text from unnecesary punctuation or urls.
    Input: raw data
    
    Output: clean_tokens - clean messages
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    #find the url in texts and replace them with space using a for loop
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    #tokenize words
    tokens = word_tokenize(text)
    #initiate a lemmatizer
    lemmatizer = WordNetLemmatizer()
    #get the cleaned lowered words inside another list using for loop   
    # usinga for loop to iterate on each token
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
             
    return clean_tokens


def build_model(clf = AdaBoostClassifier()):
    """ This function builds a model using pipeline class, and feature union that combines both 
    a custom extractor for texts and a designed classifier called 'AdaBoostClassifier' used for 
    MultiOutputClassifer.
    
    Input classifier - clf is AdaBoostClassifier for default, however it is upto the user to choose
    the classifier.
    Output - cv A pipeline for ML after performing grid search that is used to make the model better.
    
    """
        #it is not ready yet but i will include it anyway
    pipeline = Pipeline([('features', FeatureUnion([('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('nouns_verbs', VerbAndNounExtractor())
        ])),
    
            ('clf', MultiOutputClassifier(clf))])
    
    parameters = {'features__text_pipeline__vect__ngram_range': ((1,1), (1,2)),'clf__estimator__learning_rate': [0.5, 1.0]}

    cv = GridSearchCV(pipeline, param_grid=parameters)
    #we need to include parameters and grid search 
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """This function tests how well the model performs. It takes 4 inputs and one outpput.
    
    The model - is the pipeline used to build ML model
    X_test - test messages split above
    y_test - categories for the test messages
    category_names - category names for y variable
    
    Output - there is no return but it prints scores precision, 
    recall, f1-score from the classification report function 
    for each output category of the dataset. 
    In addition, it prints the overall accuracy.
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names=category_names))
    accuracy = (y_pred == Y_test).mean().mean()
    print('accuracy{}'.format(accuracy))


def save_model(model, model_filepath):
    """ This function saves the built model in order to use to perform on datasets.
    Input:
    model - ML model
    model_filepath - where the path of the ml model is saved
    
    Output - None """
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
                
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/Disaster_Response_Project.db classifier.pkl')


if __name__ == '__main__':
    main()