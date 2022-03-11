# import pandas for dataframe manipulation
import pandas as pd


def display_model_results(model_results):
    '''
    This function takes in the model_results dataframe created in the Model stage of the 
    TelCo Churn analysis project. This is a dataframe in tidy data format containing the following
    data for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)
    The function returns a pivot table of those values for easy comparison of models, metrics, and samples. 
    '''
    # create a pivot table of the model_results dataframe
    # establish columns as the model_number, with index grouped by metric_type then sample_type, and values as score
    # the aggfunc uses a lambda to return each individual score without any aggregation applied
    return model_results.pivot_table(columns='model_number', 
                                     index=('metric_type', 'sample_type'), 
                                     values='score',
                                     aggfunc=lambda x: x)
                                     

def get_best_model_results(model_results, metric_type='accuracy', n_models=3):
    '''
    This function takes in the model_results dataframe created in the Modeling stage of the 
    TelCo Churn analysis project. This is a dataframe in tidy data format containing the following
    data for each model created in the project:
    - model number
    - metric type (accuracy, precision, recall, f1 score)
    - sample type (train, validate)
    - score (the score for the given metric and sample types)

    The function identifies the {n_models} models with the highest scores for the given metric
    type, as measured on the validate sample.

    It returns a dataframe of information about those models' performance in the tidy data format
    (as described above). 

    The resulting dataframe can be fed into the display_model_results function for convenient display formatting.
    '''
    # create an array of model numbers for the best performing models
    # by filtering the model_results dataframe for only validate scores for the given metric type
    best_models = (model_results[(model_results.metric_type == metric_type) 
                               & (model_results.sample_type == 'validate')]
                                                 # sort by score value in descending order
                                                 .sort_values(by='score', 
                                                              ascending=False)
                                                 # take only the model number for the top n_models
                                                 .head(n_models)
                                                 .model_number
                                                 # and take only the values from the resulting dataframe as an array
                                                 .values)
    # create a dataframe of model_results for the models identified above
    # by filtering the model_results dataframe for only the model_numbers in the best_models array
    # TODO: make this so that it will return n_models, rather than only 3 models
    best_model_results = model_results[(model_results.model_number == best_models[0]) 
                                     | (model_results.model_number == best_models[1]) 
                                     | (model_results.model_number == best_models[2])]

    return best_model_results



def display_model_info(model_info, model_numbers):
    '''
    This function takes in a list of model numbers and displays all info from the model_info dataframe
    for only those model numbers. 
    Info in the model_info dataframe includes:
    - model_number
    - model_type (Decision Tree, Random Forest, KNN, or Logistic Regression)
    - values for any hyperparamters passed as arguments to the model's classifier function 
    '''
    # use iloc to index the model_info dataframe for model numbers (corresponds to dataframe index)
    return pd.DataFrame(model_info.iloc[model_numbers,:])
