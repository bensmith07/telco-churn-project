import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#### explore univariate categorical
def univar_categorical(train):

    # define categorical columns
    categorical_columns = list(train.dtypes[train.dtypes == 'object'].drop(columns='customer_id').index)
    categorical_columns.remove('customer_id')

    # for each of those columns
    for col in categorical_columns:
        
        # display a frequency table
        print(pd.DataFrame(train[col].value_counts())
            .rename(columns={col: f'{col}_counts'}))
        
        # display a bar plot of those frequencies
        sns.countplot(data=train,
                    x=col)
        plt.title(f'{col}_counts')
        plt.show()

#### explore univariate quantititave
def univar_quantitative(train):

    # define quantitative columns
    quantitative_columns = ['monthly_charges', 'tenure_months']

    # for each of those columns:
    for col in quantitative_columns:
            
            # display a histogram of that column's distribution
            sns.histplot(train[col], stat='proportion')
            plt.show()
            
            # display a boxplot of that column's distribution
            sns.boxplot(train[col])
            plt.show()
            
            # display the summary statistics of the distribution
            print(pd.DataFrame(train[col].describe()))


#### explore bivariate quantitative
def bivar_quantitative(train, target):

    # define quantitative columns
    quantitative_columns = ['monthly_charges', 'tenure_months']

    # for each of the quantitative columns
    for col in quantitative_columns:
        # display a barplot of that variable for customers in each category of the target variable
        sns.barplot(data=train,
                    x=target,
                    y=col)
        # add a horizontal line representing the overall mean value for that variable
        plt.axhline(train[col].mean(), 
                    ls='--', 
                    color='black')
        plt.title(col, fontsize=14)
        plt.show()

#### explore multivariate 1

def multivar_1(train, target):

    # define quantitative columns
    quantitative_columns = ['monthly_charges', 'tenure_months']

    # create and display a pairplot comparing the distributions of each quantitative variable
        # in each category of the target variable
    g = sns.pairplot(train[quantitative_columns + [target]], hue=target)
    plt.show()


### explore correlations

def multivar_absolute_correlations(train):

    # create dataframe of correlation values for target variable
    train_correlations = pd.DataFrame(train.drop(columns='customer_id').corr().enc_churn.sort_values())
    # use the absolute values of the correlations in a new dataframe
    train_corr_abs = pd.DataFrame(train_correlations.apply(lambda corr: abs(corr).sort_values(ascending=False)))

    return train_corr_abs


#### chi2_test

# defining a function to encapsulate repeated code for chi2 tests:

def chi2_test(data_for_category1, data_for_category2, alpha=.05):
    
    # create dataframe of observed values
    observed = pd.crosstab(data_for_category1, data_for_category2)
    
    # conduct test using scipy.stats.chi2_contingency() test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    # round the expected values
    expected = expected.round(1)
    
    # output
    print('Observed\n')
    print(observed.values)
    print('---\nExpected\n')
    print(expected)
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
    # evaluate the hypothesis against the established alpha value
    if p < alpha:
        print('\nReject H0')
    else: 
        print('\nFail to Reject H0')


#### when are customers most likely to churn

def when_customers_churn(train):

    for col in ['tenure_months', 'tenure_quarters', 'tenure_years']:
        print('=' * 50)
        print(col.upper())

        # display a histogram of tenure for customers who have churned
        sns.histplot(train[train.churn == 'Yes'][col], stat='proportion')
        plt.show()

        # display a boxplot of months of tenure for customers who have churned
        sns.boxplot(train[train.churn == 'Yes'][col])
        plt.show()

        # display summary statistics for months of tenure for customers who have churned
        print(pd.DataFrame(train[train.churn == 'Yes'][col].describe()))
        
        # display the most common value
        print('mode:\t\t', train[train.churn == 'Yes'][col].mode()[0])

#### when are customers most likely to churn - by contract type

def when_customers_churn_by_contract_type(train):

    # separate the data by contract_type 
    churned_1yr = train[(train.contract_type == 'One year') & (train.churn == 'Yes')]
    churned_2yr = train[(train.contract_type == 'Two year') & (train.churn == 'Yes')]
    churned_monthly = train[(train.contract_type == 'Month-to-month') & (train.churn == 'Yes')]

    # for each category of contract type
    for group in [churned_1yr, churned_2yr, churned_monthly]:
        print('=' * 50)
        print(group.contract_type.mode()[0])
        
        # display a histogram of months of tenure for that group
        sns.histplot(group.tenure_months, stat='proportion')
        plt.show()
        
        # display a boxplot of months of tenure for that group
        sns.boxplot(group.tenure_months)
        plt.show()
        
        # display summary statistics for moths of tenure for that group
        print(pd.DataFrame(group.tenure_months.describe()))
        
        # display the most common value
        print('mode:\t\t', group.tenure_months.mode()[0])



#### display churn by service type

def churn_by_int_service_type(train):

    churn_rates = (pd.DataFrame(train.groupby(by='internet_service_type').mean().enc_churn)
                .reset_index()
                .sort_values(by='enc_churn'))
    sns.barplot(data = churn_rates,
                x = 'internet_service_type', 
                y = 'enc_churn')
    plt.axhline(train.enc_churn.mean(), 
                    ls='--', 
                    color='black')
    plt.show()

#### hypothesis test 1: MANN WHITNEY U TEST FOR MONTHLY CHARGES

def hypothesis_test_monthly_charges(train):

    # get the data
    train_churned = train[train.churn == 'Yes']
    train_not_churned = train[train.churn == 'No']

    # Mann Whitney U test
    # conduct the test
    u, p = stats.mannwhitneyu(train_churned.monthly_charges, train_not_churned.monthly_charges, alternative='greater')

    # display test info and results
    print(f'MANN-WHITNEY U TEST FOR: MONTHLY_CHARGES')
    print()
    print(f'u = {u}')
    print(f'p = {p.round(4)}')
    print()