import pandas as pd

def display_uniques(df, n_unique_limit=10):
    '''
    This function takes in a dataframe and displays the unique values for each column in the form of a pandas 
    dataframe. Intended to display unique values for categorical variables. 
    
    n_unique_limit - default 10 - establishes the max number of unique valuescontained in the columns we 
    want to display.
    '''
    # create empty df
    newdf = pd.DataFrame()
    
    # for each column in the df
    for col in df.columns:
        # create a column in the new df that contains only the unique values from that column in the original df
        newdf[col] = pd.Series(df[col].unique())
    
    # drop all columns that have a number of unique values that is greater than our established limit
    newdf = newdf.drop(columns=newdf.columns[newdf.count() > n_unique_limit])
    
    # truncate the dataframe to the appropriate number of rows
    newdf = newdf.head(newdf.count().max())
    
    # fill nulls with empty strings for a cleaner display
    newdf = newdf.fillna('')
    
    return newdf

def display_uniques_1(df, limit=10):
    for col in df.columns[df.nunique() < limit]:
        print(f'Column: {col}\nUnique Values: {df[col].unique()}')
        print('----------')

def display_uniques_2(df, limit=10):
    for col in df.columns:
        print(f'Column: {col}')
        if df[col].nunique() < 10:
            print(f'Unique Values: {df[col].unique()}')
        else: 
            print(f'Range of Values: [{df[col].min()} - {df[col].max()}]')
        print('-----------------------')