import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import pos_tag, ne_chunk
from nltk.tokenize import sent_tokenize
from sklearn.ensemble import GradientBoostingClassifier


def load_data(database_filepath):
    '''
    INPUT
    
    file path to the SQLite database
    
    OUTPUT
    
    X, Y, category_names: list of messages to categorize, list of categorization labels, names of categories to be predicted
    
    '''
    # create engine to connect to database
    engine = create_engine('sqlite:///' + database_filepath)
    # read data from database
    df = pd.read_sql_table('DisasterResponseTable',engine)
    # define X and Y
    X= df['message']
    Y= df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names
    

def tokenize(text):
    '''
    INPUT
    
    list of texts to tokenize
    
    OUTPUT
    
    a tokenized list of text
    
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    #Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)
    
    #Remove extra space within a meassage 
    text = " ".join(text.split())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    # remove stop words
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]
    
    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class, as in the course material, extracts the starting verb of a sentence
    """
    def starting_verb(self, text):
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        #attach POS tags to sentences 
        #try/except used because pos_tags[0] raises out of range exception although our data has no nulls)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            except: pass
        return False
                       
                
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    

def build_model():
    '''
    INPUT
    
    No input
    
    OUTPUT
    
    NLP Grid search pipeline model 
    
    '''
    pipeline_boost = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('moclf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
    'moclf__estimator__learning_rate' : [0.1,1],
    'moclf__estimator__n_estimators': [50, 100]}
    
    cv2 = GridSearchCV(pipeline_boost, param_grid=parameters, verbose = 2) #, n_jobs=-1) 
 
    return cv2


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model: pipeline to get predictions
    X_test: to feed to the model
    Y_test: to evaluate the model against Y_pred
    category_names: to evaluate each category 
    
    OUTPUT
    
    prints a classification report of the evaluation  
    
    '''
    
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    INPUT
    model: final pipeline to save
    model_filepath: path to save pickle file
    
    OUTPUT
    
    prints a classification report of the evaluation  
    
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()