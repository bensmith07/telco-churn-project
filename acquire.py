import os
import pandas as pd
from env import get_db_url, user, password, host

# define a function for obtaining Telco Customer Data
def get_telco_data():
    '''
    This function acquires TelCo customer data from the Codeup MySQL database, or from a .csv 
    file in the local directory. If a local .csv exists, the .csv is imported using pandas. If
    there is no local file, data.codeup.com is accessed with the appropriate credentials, and the 
    data is obtained via SQL query and then imported via pandas. The SQL query joins all necessary
    tables for customer data. After obtaining from the database, the pandas dataframe is cached to 
    a local CSV for future use. The data is returned from the function as a pandas dataframe. 
    '''
    
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