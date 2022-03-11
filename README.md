# Predicting Customer Churn at Telco

## About the Project 

We will conduct an in depth analysis of customer data for the hypothetical telecommunications company, TelCo. We will use exploratory analysis techniques to identify the key drivers of customer churn, then use machine learning algorithms to create a model capable of predicting whether or not a customer will churn. 

### Project Goals

By identifying key drivers of customer churn and creating a predictive model, the company can focus resources on those customers most likey to churn, proactively engaging them in ways that make them more likely to continue using Telco services. 

### Project Description

Telco aims to provide an excellent product at a fair price, and to maximize customer satisfaction at every opportunity. A key metric of success in these goals is the customer churn rate. To date, Telco customers have been leaving the company at an unacceptable rate. To maintain the company's reputation as the best in the business, we need to do something about this. 

We will analyze the attributes of customers who have been more or less likely to churn, develop a model for predicting churn based on those atributes, and leave with both recommendations for future passengers and predictions of churn for a list of current customers (delivered via .csv)


### Initial Questions

- Does demographic information affect churn rate? If so, improvements aimed at better engaging specific demographic populations may be warranted. 

- Do customers who churn have higher average monthly charges than those who don't? If so, price incentives may be an effective retention strategy. 

- What month (or range of months) are customers most likely to churn? Optimizing for customer satisfaction in these months might get them "over the hump" and turn them into long-term loyal customers. 

  - Does this change when controlling for contract type?

- Is there a service associated with more churn than expected? If so, further investigation might uncover dissatisfaction with the service itself. 


### Data Dictionary

| Variable          | Meaning                                                   | values          |
| -----------       | -----------                                               | -----------     |
| churn             | whether a customer has left the company (target variable) | Yes, No
| gender            | customer gender                                         | male, female    |
| senior_citizen    | whether a customer is a senior citizen                  | 0 (no), 1 (yes) |
| partner           | whether a customer has a partner                        | Yes, No         |
| dependents        | whether a customer has dependents                       | Yes, No         |
| tenure_months     | # of months a customer has been with Telco (truncated)  | 1 - 72          |
| tenure_quarters   | customer tenure in quarters (rounded up)                | 1 - 24          |
| tenure_years      | customer tenure in years (rounded up)                   | 1 - 6           |
| phone_service     | whether a customer subscribes to phone service          | Yes, No                           |
| multiple_lines    | whether a customer subscribes to multiple phone lines   | Yes, No, No phone service         |
| online_security   | whether a customer subscribes to online security  | Yes, No, No internet service |
| online_backup     | whether a customer subscribes to online backup  | Yes, No, No internet service  |
| device_protection | whether a customer subscribes to device protection  | Yes, No, No internet service  |
| tech_support      | whether a customer subscribes to tech support | Yes, No, No internet service  |
| streaming_tv      | whether a customer subscribes to television streaming | Yes, No, No internet service  |
| streaming_movies  | whether a customer subscribes to movie streaming services | Yes, No, No internet service  |
| paperless_billing | whether a customer has opted in to paperless billing  | Yes, No |
| monthly_charges   | total amount in dollars the customer pays each month  | 18.25 - 118.75  |
| contract_type     | duration of customer's current contract | Month-to-month, One-year, Two-year  |
| internet_service_type | type of internet service to which a customer subscribes | DSL, Fiber optic, No internet service |
| payment_type  | method of the customer's monthly payment  | Mailed check, Electronic check, Credit card (automatic), Bank transfer (automatic)  |


### Steps to Reproduce

1. You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the titanic_db.passengers table. The env.py should also contain a function named get_db_url() that establishes the string value of the database url. Store that env file locally in the repository. 
2. clone my repo (including the acquire.py, prepare.py, explore.py, and model.py modules) (confirm .gitignore is hiding your env.py file)
3. libraries used are pandas, matplotlib, seaborn, numpy, sklearn. 

### The Plan

1. Acquisition
- In this stage, I obtained TelCo customer data by querying the Codeup MySQL database hosted at data.codeup.com.
2. Preparation
- I cleaned and prepped the data by:
  - removing duplicate records and removing records of brand new customers
  - modifying records to fit the appropriate data types 
  - removing unnecessary or unhelpful features and engineering new features where appropriate
3. Exploration
- I conducted an initial exploration of the data
- then explored further, to answer the initial questions posed above
4. Modeling 
- Using varying parameter values and combinations of features, I tested over [_____________] different models of varying types, including:
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbor
  - Logistic Regression
- I then chose the model which performed with the highest accuracy on unseen data. and engineering new features where appropriate

### How did we do?

We have succeeded in identifying some features that are predictors of churn. We set out to use machine learning to create a model that predicts customer churn, and we were able to predict churn with 77.4% accuracy, which is approximately a 4% improvement over our baseline predictions. 

### Key Findings

Some customer attributes that increase their probability of churn include:
- being earlier in their tenure, in particular in approximately their first year
- having higher monthly charges
- being a senior citizen
- having a partner and dependants

### Recommendations

Since higher monthly charges and early tenure are major drivers of churn, consider offering discounted reates in the first year of tenure. 

### Next Steps

Given more time, I would test models using additional combinations of features. Given enough time and computational resources, I would like to run each of the model types above using all possible combinations of features. 

Additionally, further exploration into those features found to have the largest effect on the model could prove useful.