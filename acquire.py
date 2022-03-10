import os
import pandas as pd
from env import get_db_url, user, password, host

# define a function for obtaining Telco Customer Data
def get_telco_data():
    
    filename = 'telco_churn.csv'
    
    # check for existing csv file in local directory
    # if it exists, return it as a datframe
    if os.path.exists(filename):
        print('Reading from local CSV...')
        return pd.read_csv(filename)
    
    # if no local directory exists, query the codeup SQL database 
    
    # utilize function defined in env.py to define the url
    url = get_db_url('telco_churn')
    
    # join the customer, contract_types, internet_service_types, and payment_types tables
    sql = '''
    SELECT * 
      FROM customers
        JOIN contract_types USING(contract_type_id)
        JOIN internet_service_types USING(internet_service_type_id)
        JOIN payment_types USING(payment_type_id)
    '''
    
    # return  the results of the query as a dataframe
    print('No local file exists\nReading from SQL database...')
    df = pd.read_sql(sql, url)
    
    # save the dataframe to the local directory as a CSV for future ease of access
    print('Saving to local CSV...')
    df.to_csv(filename, index=False)
    
    # return the dataframe
    return df