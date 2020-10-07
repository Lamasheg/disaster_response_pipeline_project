import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    
    file paths to the messages and categories csv files
    
    OUTPUT
    
    df: a pandas dataframe that is a merge of messages and categories
    
    '''
    messages = pd.read_csv(messages_filepath)
    categories =  pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    '''
    INPUT
    
    df: a pandas dataframe
    
    OUTPUT
    
    df: a pandas dataframe that is a clean version of the original df
    
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand= True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[[0]]
    # use this row to extract a list of new column names for categories
    category_colnames = [name.split('-')[0] for name in row.values[0]]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #keep only the last character of each string (the 1 or 0)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop(['categories'],axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # set ['related'] column values > 1 to 1
    df.loc[df['related'] > 1,'related'] = 1
    # drop child alone column 
    df = df.drop(['child_alone'], axis = 1)
    
    return df
    




def save_data(df, database_filename):
    '''
    Save the clean df to database
    
    INPUT
    
    df: The clean version of our pandas dataframe
    database_filename: the path to SQLite database
    
    OUTPUT
    
    df: a pandas dataframe that is a clean version of the original df
    
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponseTable', engine, index=False,if_exists='replace') 
    


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
              'DisasterResponse.db')


if __name__ == '__main__':
    main()