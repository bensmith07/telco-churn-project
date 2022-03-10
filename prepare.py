import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def train_test_validate_split(df, target, test_size=.2, validate_size=.3, random_state=42):
    
    train, test = train_test_split(df, test_size=.2, random_state=42, stratify=df[target])
    train, validate = train_test_split(train, test_size=.3, random_state=42, stratify=train[target])
    
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')
    
    return train, test, validate

def prep_iris(df):
    df = df.drop(columns=['species_id', 'measurement_id'])
    df = df.rename(columns={'species_name': 'species'})
    dummy_df = pd.get_dummies(df['species'], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)    
    return df

def prep_titanic(df, drop_after_encoding=True):
    # drop duplicate rows, if they exist:
    df = df.drop_duplicates()
    # drop rows where age is null
    df = df[df.age.notna()]
    # drop unnecessary columns
    df = df.drop(columns=['class', 'embarked', 'deck', 'passenger_id'])
    # rename columns
    df = df.rename(columns={'parch': 'n_parents_and_children', 'sibsp': 'n_sibs_and_spouse'})
    # add family size column
    df['family_size'] = df.n_parents_and_children + df.n_sibs_and_spouse
    # encode categorical columbns with dummy variables then drop the original columns
    categorical_columns = ['sex', 'embark_town']
    for col in categorical_columns:
        dummy_df = pd.get_dummies(df[col],
                                  prefix=df[col].name,
                                  drop_first=True,
                                  dummy_na=False)
        df = pd.concat([df, dummy_df], axis=1)
        if drop_after_encoding:
            df = df.drop(columns=col)
    return df

def prep_telco(df):
    
    # drop duplicate rows, if present
    
    df = df.drop_duplicates()
    
    # clean up total charges column and cast as float
    df['total_charges'] = df.total_charges.replace(' ', np.nan).astype(float)
    
    # removing brand new customers
    df = df[df.tenure != 0]
    
    # drop columns:
    
    # *_type_id columns are simply foreign key columns that have corresponding string values
    # customer_id is a primary key that is not useful for our analysis
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'])
    
    # encode categorical columns with dummy variables
    
    categorical_columns = list(df.drop(columns='churn').dtypes[df.drop(columns='churn').dtypes == 'object'].index)
    
    for col in categorical_columns:
        dummy_df = pd.get_dummies(df[col],
                                  prefix=df[col].name,
                                  drop_first=True,
                                  dummy_na=False)
        df = pd.concat([df, dummy_df], axis=1)
        df = df.drop(columns=col)
        
    return df



#### AUSTIN ANIMAL CENTER DATA SET FUNCTIONS ####

def aac_prep(intakes, outcomes):

    # rename columns

    for col in intakes.columns:
        intakes = intakes.rename(columns={col: f'{col.lower().replace(" ", "_")}'})

    for col in outcomes.columns:
        outcomes = outcomes.rename(columns={col: f'{col.lower().replace(" ", "_")}'})

    cols = ['name', 'datetime', 'monthyear', 'animal_type', 'breed', 'color']
    for col in cols:
        intakes = intakes.rename(columns={col: col+'_intake'})
        outcomes = outcomes.rename(columns={col: col+'_outcome'})
        
    
    # drop animals that have more than one entry (animals that were taken in on more than one occasion) 
    # (we should remove this step in later analysis)
    
    intakes_duplicated = intakes[intakes.animal_id.duplicated(keep=False)].sort_values('animal_id')
    outcomes_duplicated = outcomes[outcomes.animal_id.duplicated(keep=False)].sort_values('animal_id')
    
    intakes = intakes.loc[~ intakes.animal_id.isin(intakes_duplicated.animal_id)]
    outcomes = outcomes.loc[~ outcomes.animal_id.isin(outcomes_duplicated.animal_id)]
    
    
    # drop outcomes that don't have a corresponding intake, and vis-versa
    
    outcomes_without_intake = outcomes.loc[~ outcomes.animal_id.isin(intakes.animal_id)]
    outcomes = outcomes.loc[~ outcomes.animal_id.isin(outcomes_without_intake.animal_id)]
    
    intakes_without_outcomes = intakes.loc[~ intakes.animal_id.isin(outcomes.animal_id)]
    intakes = intakes.loc[~ intakes.animal_id.isin(intakes_without_outcomes.animal_id)]
    
    
    # join the dataframes
    
    df = pd.merge(intakes, outcomes, on='animal_id')
    
    
    # drop variables from the original outcomes table (since by definition, they're not drivers of outcome)
    
    columns = ['datetime_outcome', 'monthyear_outcome', 'date_of_birth', 'outcome_subtype', 'animal_type_outcome', 'sex_upon_outcome', 'age_upon_outcome', 'breed_outcome', 'color_outcome', 'name_outcome']
    df = df.drop(columns=columns)
    
    
    # split the month_year column to extract the month (proxy for time of year), then drop month_year column
    
    df['month_intake'] = df.monthyear_intake.str.split().apply(lambda row: row[0])
    df = df.drop(columns='monthyear_intake')
    
    
    # split the sex_upon_intake column into fixed = True/False and sex = male/female
    # then drop the sex_upon_intake column
    
    df['fixed'] = df.sex_upon_intake.map({'Neutered Male': True,
                                      'Spayed Female': True,
                                      'Intact Male': False,
                                      'Intact Female': False,
                                      'Unknown': 'unknown'})
    df['sex'] = df.sex_upon_intake.map({'Neutered Male': 'male',
                                      'Spayed Female': 'female',
                                      'Intact Male': 'male',
                                      'Intact Female': 'female',
                                      'Unknown': 'unknown'})
    df = df.drop(columns='sex_upon_intake')
    
    
    # rename columns
    
    df = df.rename(columns={'animal_type_intake': 'animal_type',
                        'breed_intake': 'breed', 
                        'color_intake': 'color',
                        'name_intake': 'name'})
    
    # determine if breed is 'mixed'
        # based on whether the breed description contains the word "Mix"
        # and based on whether there is more than one breed listed in the description (separated by "/")
    # create new column breed_mixed = True/False
    # then remove the word "Mix" from the breed description
    
    def check_mixed(breed):
        if 'Mix' in breed or '/' in breed:
            return True
        else:
            return False

    df['breed_mixed'] = df.breed.apply(lambda row: check_mixed(row))
    df['breed'] = df.breed.str.replace(' Mix', '')

    # split the breed description into multiple columns when there is more than one listed
    # then drop the original breed column
    
    def breed_split_1(breed):
        if len(breed.split('/')) == 1:
            return breed
        else:
            return breed.split('/')[0]

    def breed_split_2(breed):
        if len(breed.split('/')) > 1:
            return breed.split('/')[1]
        else:
            return np.nan

    def breed_split_3(breed):
        if len(breed.split('/')) > 2:
            return breed.split('/')[2]
        else:
            return np.nan

    df['breed_1'] = df.breed.apply(breed_split_1)
    df['breed_2'] = df.breed.apply(breed_split_2)
    df['breed_3'] = df.breed.apply(breed_split_3)
    df = df.drop(columns='breed')
    
    
    # split the color descriptino into multiple columns when there is more than one listed
    # then drop the original color column
    
    def color_split_1(color):
        if len(color.split('/')) == 1:
            return color
        else:
            return color.split('/')[0]

    def color_split_2(color):
        if len(color.split('/')) > 1:
            return color.split('/')[1]
        else:
            return np.nan

    df['color_1'] = df.color.apply(color_split_1)
    df['color_2'] = df.color.apply(color_split_2)
    df = df.drop(columns='color')
    
    
    # convert age column into pandas timedelta (number of days)
    
    df['age_number'] = df.age_upon_intake.str.split().apply(lambda row: int(row[0]))
    df['age_units'] = df.age_upon_intake.str.split().apply(lambda row: row[1])
    df['age_multiplier'] = df.age_units.map({'day': 1, 
                                             'days': 1, 
                                             'week': 7, 
                                             'weeks': 7,
                                             'month': 30, 
                                             'months': 30, 
                                             'year': 365, 
                                             'years': 365})
    df['age_intake'] = df.age_number * df.age_multiplier
    df['age_intake'] = df.age_intake.apply(lambda row: pd.Timedelta(days=row))
    df = df.drop(columns=['age_number', 'age_units', 'age_multiplier', 'age_upon_intake'])
    
    # convert the date & time of the intake into a pandas datetime type
    
    df['datetime_intake'] = pd.to_datetime(df.datetime_intake)
    
    return df

def aac_get_dogs(df):
    df = df[df.animal_type == 'Dog']

    return df

def aac_prep_for_modeling(df):
    # drop columns not used for modeling at this time
    df = df.drop(columns=['datetime_intake', 'found_location', 'name', 'animal_id', 'breed_2', 'breed_3', 'color_2', 'name'])
    # drop rows in order to focus on most common outcome types
    df = df[df.outcome_type.isin(['Adoption', 'Transfer', 'Return to Owner'])]
    # columns to hot code
    categorical_columns = ['fixed', 'breed_mixed', 'intake_type', 'intake_condition', 'animal_type', 'month_intake', 'sex', 'breed_1', 'color_1']
    # hot coding dummy variables
    for col in categorical_columns:
        dummy_df = pd.get_dummies(df[col],
                                  prefix=df[col].name,
                                  drop_first=True,
                                  dummy_na=False)
        df = pd.concat([df, dummy_df], axis=1)
        # drop original column
        df = df.drop(columns=col)
    # turn age_intake timedelta into float
    df['age_intake'] = df.age_intake / pd.Timedelta(days=1)
    return df