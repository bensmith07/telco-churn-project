import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

#### prep 1

def prep_telco_1(df):

    # drop duplicate rows, if present
    df = df.drop_duplicates()

    # clean up total_charges column and cast as float
    df['total_charges'] = df.total_charges.replace(' ', np.nan).astype(float)

    # drop rows with any null values
    df = df.dropna()

    # removing brand new customers
    df = df[df.tenure != 0]

    # drop any unnecessary, unhelpful, or duplicated columns. 
    # type_id columns are simply foreign key columns that have corresponding string values
    # customer_id is a primary key that is not useful for our analysis
    # total_charges is essentially a function of monthly_charges * tenure
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'total_charges'])

    # change senior citizen to object types Yes/No for exploration purposes
    df['senior_citizen'] = df.senior_citizen.map({1: 'Yes', 0: 'No'})

    # add a feature: tenure by quarters
    df['tenure_quarters'] = df.tenure.apply(lambda months: math.ceil(months / 3))

    # add a feature: tenure by years
    df['tenure_years'] = df.tenure.apply(lambda months: math.ceil(months / 12))

    # rename tenure columns
    df = df.rename(columns={'tenure': 'tenure_months'})

    return df


#### prep 2

def prep_telco_2(df):

    # define categorical columns
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].drop(columns='customer_id').index)
    categorical_columns.remove('customer_id')

    # one-hot encoding those columns
    for col in categorical_columns:
        dummy_df = pd.get_dummies(df[col],
                                prefix=f'enc_{df[col].name}',
                                drop_first=True,
                                dummy_na=False)
        
        # add the columns to the dataframe
        df = pd.concat([df, dummy_df], axis=1)
        
    # clean up the column names
    for col in df.columns:
        df = df.rename(columns={col: col.lower()})
        df = df.rename(columns={col: col.replace(' ', '_')})
    df = df.rename(columns={'enc_churn_yes': 'enc_churn'})

    return df


#### train, test, validate split

def train_test_validate_split(df, target, test_size=.2, validate_size=.3, random_state=42):
    
    train, test = train_test_split(df, test_size=.2, random_state=42, stratify=df[target])
    train, validate = train_test_split(train, test_size=.3, random_state=42, stratify=train[target])
    
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')
    
    return train, test, validate