import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def univar_categorical(train):
    '''
    This takes in a dataframe representing a train sample and displays frequency tables and barplots 
    for each categorical variable in the Telco Customer dataset. 
    '''
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

def univar_quantitative(train):
    '''
    This function takes in the train sample dataframe from the Telco customer dataset and 
    displays a histogram, a boxplot, and summary statistics for the distribution of each of 
    the quantitative variables. 
    '''
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

def bivar_quantitative(train, target):
    '''
    This function takes in the train sample dataframe from the Telco customer dataset and displays
    a barplot of the distribution of each quantitative variable, grouped by each category of the 
    target variable, with a horizontal line representing the overall average value. 
    '''
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

def multivar_1(train, target):
    '''
    This function takes in the train sample dataframe from the Telco customer dataset and a target variable lable,
    then displays a seaborn pairplot using each of the quantitative variables in the dataset. 
    '''
    # define quantitative columns
    quantitative_columns = ['monthly_charges', 'tenure_months']

    # create and display a pairplot comparing the distributions of each quantitative variable
        # in each category of the target variable
    g = sns.pairplot(train[quantitative_columns + [target]], hue=target)
    plt.show()

def multivar_absolute_correlations(train):
    '''
    This function takes in the train sample dataframe from the Telco customer dataset then returns a dataframe 
    showing the absolute correlation values of each variable with the target variable churn. The dataframe is 
    returned, rather than printed, so that it can be displayed with visually appealing format in jupyter notebook
    by calling the function without a print statement or other assignment. 
    '''
    # create dataframe of correlation values for target variable
    train_correlations = pd.DataFrame(train.drop(columns='customer_id').corr().enc_churn.sort_values())
    # use the absolute values of the correlations in a new dataframe
    train_corr_abs = pd.DataFrame(train_correlations.apply(lambda corr: abs(corr).sort_values(ascending=False)))
    # rename columns and features index for display
    train_corr_abs.columns = ['Correlation']
    train_corr_abs.index = ['Churn', 
                            'Tenure in Months', 
                            'Tenure Quarters', 
                            'Tenure Years', 
                            'Payment Type (Electronic Check)', 
                            'Internet Service Type (Fiber Optic)', 
                            'Contract Type (Two-Year)', 
                            'Internet Service Type (None)', 
                            'Streaming Movies (No Internet Service)',
                            'Streaming TV (No Internet Service)', 
                            'Tech Support (No Internet Service)', 
                            'Online Backup (No Internet Service)',
                            'Online Security (No Internet Service)', 
                            'Device Protection (No Internet Service)',
                            'Monthly Charges', 'Contract Type (One Year)', 
                            'Paperless Billing',
                            'Tech Support', 
                            'Senior Citizen', 
                            'Partner', 
                            'Dependents', 
                            'Online Security', 
                            'Payment Type (Credit Card - Automatic)', 
                            'Payment Type (Mailed Check)', 
                            'Online Backup', 
                            'Streaming TV', 
                            'Device Protection', 
                            'Streaming Movies', 
                            'Multiple Lines', 
                            'Multiple Lines (No Phone Service)',
                            'Phone Service', 
                            'Gender']
    return train_corr_abs

def chi2_test(data_for_category1, data_for_category2, alpha=.05):
    '''
    This function takes in two array-like objects of categorical data (two columns
    of a pandas dataframe is the intended use case) and uses scipy.stats to 
    conduct a chi^2 test for independence. 
    It prints crosstabs of the observed and expected values, and determines whether to 
    reject the null hypothesis based on a given value of alpha. 
    '''
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

def when_customers_churn(train):
    '''
    This function takes in the train sample dataframe, filters for only customers who have churned, 
    and displays information about tenure distribution in months, quarters, and years, 
    using histograms, boxplots, and summary statistics.
    '''
    for col in ['tenure_months', 'tenure_quarters', 'tenure_years']:
        print('=' * 50)
        print('DISTRIBUTION OF ', col.upper())

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
    '''
    This function takes in the train dataframe sample, separates the sample by contract type, filtering
    for only customers who have churned, then displays information about the distribution of tenure in 
    months for each contract type.
    '''

    # separate the data by contract_type 
    churned_1yr = train[(train.contract_type == 'One year') & (train.churn == 'Yes')]
    churned_2yr = train[(train.contract_type == 'Two year') & (train.churn == 'Yes')]
    churned_monthly = train[(train.contract_type == 'Month-to-month') & (train.churn == 'Yes')]

    # for each category of contract type
    for group in [churned_1yr, churned_2yr, churned_monthly]:
        print('=' * 50)
        print('DISTRIBUTION OF CHURN FOR ', group.contract_type.mode()[0].upper(), ' CONTRACT TYPES')
        
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
    plt.title('Churn by Internet Service Type')
    plt.show()

#### hypothesis test 1: MANN WHITNEY U TEST FOR MONTHLY CHARGES

def hypothesis_test_monthly_charges(train):
    '''
    This function takes in the train dataset, then runs a Mann-Whitney U test to test average
    monthly charges for customers who have churned against average monthly charges for customers
    who have not churned. It then displays the test statistic and p-alue. 
    '''
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


def plot_overall_churn(df):
    '''
    This function displays a pie chart created in matplotlib.pyplot, showing
    the overall percentage of customers who have churned vs. those who have not. 
    '''
    # create the pie chart, using a count (len()) of all customers who have churned
    # and a count of those who have not
    plt.pie([len(df[df.churn == 'Yes']), len(df[df.churn == 'No'])], 
            # add a label for the churn == Yes category
            labels=['Churned', None], 
            # specify slice colors
            colors=['red', 'lightblue'],
            # specify percentage formatting
            autopct='%.0f%%')
    # display the chart
    plt.title('Overall Churn Rate')
    plt.show()


def churn_rate_by_gender(train):
    '''
    This function takes in the train sample dataframe, then creates a dataframe that includes
    the rate of churn for each gender of customer, and displays those rates in comparison to 
    each other using a seaborn barplot, with the overall average churn rate represented as a 
    dashed horizontal line. 
    '''
    churn_rates = (pd.DataFrame(train.groupby(by='gender').mean().enc_churn)
                   .reset_index()
                   .sort_values(by='enc_churn'))
    sns.barplot(data = churn_rates,
                x = 'gender', 
                y = 'enc_churn')
    plt.axhline(train.enc_churn.mean(), 
                    ls='--', 
                    color='black')
    plt.title('Churn Rate by Gender')
    plt.show()


def churn_rate_by_senior(train):
    '''
    This function takes in the train sample dataframe, then creates a dataframe that includes
    the rate of churn for each senior citizen status, and displays those rates in comparison to 
    each other using a seaborn barplot, with the overall average churn rate represented as a 
    dashed horizontal line. 
    '''
    churn_rates = (pd.DataFrame(train.groupby(by='senior_citizen').mean().enc_churn)
                   .reset_index()
                   .sort_values(by='enc_churn'))
    sns.barplot(data = churn_rates,
                x = 'senior_citizen', 
                y = 'enc_churn')
    plt.axhline(train.enc_churn.mean(), 
                    ls='--', 
                    color='black')
    plt.title('Churn Rate by Senior Citizen Status')
    plt.show()


def churn_rate_by_partner(train):
    '''This function takes in the train sample dataframe, then creates a dataframe that includes
    the rate of churn for each partner status, and displays those rates in comparison to 
    each other using a seaborn barplot, with the overall average churn rate represented as a 
    dashed horizontal line. 
    '''
    churn_rates = (pd.DataFrame(train.groupby(by='partner').mean().enc_churn)
                   .reset_index()
                   .sort_values(by='enc_churn'))
    sns.barplot(data = churn_rates,
                x = 'partner', 
                y = 'enc_churn')
    plt.axhline(train.enc_churn.mean(), 
                    ls='--', 
                    color='black')
    plt.title('Churn Rate by Partner Status')
    plt.show()

def churn_rate_by_dependents(train):
    '''This function takes in the train sample dataframe, then creates a dataframe that includes
    the rate of churn for each status of dependants, and displays those rates in comparison to 
    each other using a seaborn barplot, with the overall average churn rate represented as a 
    dashed horizontal line. 
    '''
    churn_rates = (pd.DataFrame(train.groupby(by='dependents').mean().enc_churn)
                    .reset_index()
                    .sort_values(by='enc_churn'))
    sns.barplot(data = churn_rates,
                x = 'dependents', 
                y = 'enc_churn')
    plt.axhline(train.enc_churn.mean(), 
                    ls='--', 
                    color='black')
    plt.title('Churn Rate by Dependents Status')
    plt.show()

def monthly_charges_by_churn(train):
    '''
    This function takes in the train sample dataframe, creates a dataframe that includes 
    average monthly_charges for customers who have churned and average monthly charges for 
    customers who have not churned, then displays those averages in comparison to each other
    using a seaborn barplot, with the overall average charges represented by a dashed horizontal line. 
    '''
    charges = (pd.DataFrame(train.groupby(by='churn').monthly_charges.mean())
                   .reset_index())
    sns.barplot(data = charges,
                x = 'churn', 
                y = 'monthly_charges')
    plt.axhline(train.monthly_charges.mean(), 
                    ls='--', 
                    color='black')
    plt.title('Average Monthly Charges by Churn Status')
    plt.show()