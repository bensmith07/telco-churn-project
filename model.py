# import pandas for dataframe manipulation
import pandas as pd
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


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

def run_baseline(train,
                 validate,
                 target,
                 positive,
                 model_number,
                 model_info,
                 model_results):
    '''
    This function takes in the train and validate samples as dataframes, the target variable label, the positive condition label,
    an initialized model_number variable, as well as model_info and model_results dataframes dataframes that will be used for 
    storing information about the models. It then performs the operations necessary for making baseline predictions
    on our dataset, and stores information about our baseline model in the model_info and model_results dataframes. 
    The model_number, model_info, and model_results variables are returned (in that order). 
    '''

    # separate each sample into x (features) and y (target)
    x_train = train.drop(columns=target)
    y_train = train[target]

    x_validate = validate.drop(columns=target)
    y_validate = validate[target]


    # store baseline metrics

    # identify model number
    model_number = 'baseline'
    #identify model type
    model_type = 'baseline'

    # store info about the model

    # create a dictionary containing model number and model type
    dct = {'model_number': model_number,
           'model_type': model_type}
    # append that dictionary to the model_info dataframe
    model_info = model_info.append(dct, ignore_index=True)

    # establish baseline predictions for train sample
    y_pred = baseline_pred = pd.Series([train[target].mode()[0]]).repeat(len(train))

    # get metrics

    # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
    dct = {'model_number': model_number, 
           'sample_type': 'train', 
           'metric_type': 'accuracy',
           'score': sk.metrics.accuracy_score(y_train, y_pred)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'sample_type': 'train', 
           'metric_type': 'precision',
           'score': sk.metrics.precision_score(y_train, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'sample_type': 'train', 
           'metric_type': 'recall',
           'score': sk.metrics.recall_score(y_train, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'sample_type': 'train', 
           'metric_type': 'f1_score',
           'score': sk.metrics.f1_score(y_train, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    # establish baseline predictions for validate sample
    y_pred = baseline_pred = pd.Series([train[target].mode()[0]]).repeat(len(validate))

    # get metrics

    # create dictionaries for each metric type for the validate sample and append those dictionaries to the model_results dataframe
    dct = {'model_number': model_number, 
           'sample_type': 'validate', 
           'metric_type': 'f1_score',
           'score': sk.metrics.f1_score(y_validate, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'sample_type': 'validate', 
           'metric_type': 'accuracy',
           'score': sk.metrics.accuracy_score(y_validate, y_pred)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'sample_type': 'validate', 
           'metric_type': 'precision',
           'score': sk.metrics.precision_score(y_validate, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    dct = {'model_number': model_number, 
           'sample_type': 'validate', 
           'metric_type': 'recall',
           'score': sk.metrics.recall_score(y_validate, y_pred, pos_label=positive)}
    model_results = model_results.append(dct, ignore_index=True)

    # set the model number to from 'baseline' to 0 
    model_number = 0
    
    return model_number, model_info, model_results


def run_decision_tree(train,
                      validate,
                      target,
                      positive,
                      feature_combos,
                      model_number,
                      model_info,
                      model_results):
    '''
    This function takes in the train and validate samples as dataframes, the target variable label, the positive condition label,
    a list of feature combinations to be tested, an initialized model_number variable, as well as model_info and model_results 
    dataframes that will be used for storing information about the models. 
    
    It then performs the operations necessary
    for creating, fitting, and making predictions with various decision tree models, using each of our list of feature combinations
    and varied values for the max-depth hyperparameter. Information about these models and their performance metrics are stored in the 
    appropriate dataframes. 
    
    model_number, model_info, and model_results variables are returned (in that order). 
    '''

    # iterate over each set of features
    for features in feature_combos:
        # iterate over integers 1-10 for changing max-depth value
        for max_depth in range(1, 11):

            # create a new model number by adding 1 to the previous model number
            model_number += 1
            # establish the model type
            model_type = 'decision tree'

            # store info about the model

            # create a dictionary containing the features and hyperparamters used in this model instance
            dct = {'model_number': model_number,
                   'model_type': model_type,
                   'features': features,
                   'max_depth': max_depth}
            # append that dictionary to the model_info dataframe
            model_info = model_info.append(dct, ignore_index=True)

            # separate each sample into x (features) and y (target)
            x_train = train[features]
            y_train = train[target]

            x_validate = validate[features]
            y_validate = validate[target]


            # create the classifer

            # establish a decision tree classifier with the given max depth
            # set a random state for repoduceability
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

            # fit the classifier to the training data
            clf = clf.fit(x_train, y_train)


            # create prediction results for the model's performance on the train sample
            y_pred = clf.predict(x_train)
            sample_type = 'train'


            # get metrics

            # create dictionaries for each metric type for the train sample and append those dictionaries to the model_results dataframe
            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'accuracy',
                   'score': sk.metrics.accuracy_score(y_train, y_pred)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'precision',
                   'score': sk.metrics.precision_score(y_train, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'recall',
                   'score': sk.metrics.recall_score(y_train, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'f1_score',
                   'score': sk.metrics.f1_score(y_train, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)


            # create prediction results for the model's performance on the validate sample
            y_pred = clf.predict(x_validate)
            sample_type = 'validate'

            # get metrics

            # create dictionaries for each metric type for the validate sample and append those dictionaries to the model_results dataframe
            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'f1_score',
                   'score': sk.metrics.f1_score(y_validate, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'accuracy',
                   'score': sk.metrics.accuracy_score(y_validate, y_pred)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'precision',
                   'score': sk.metrics.precision_score(y_validate, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                   'sample_type': sample_type, 
                   'metric_type': 'recall',
                   'score': sk.metrics.recall_score(y_validate, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)     
        
        return model_number, model_info, model_results


def run_random_forest(train,
                      validate,
                      target,
                      positive,
                      feature_combos,
                      model_number,
                      model_info,
                      model_results):
    '''
    This function takes in the train and validate samples as dataframes, the target variable label, the positive condition label,
    a list of feature combinations to be tested, an initialized model_number variable, as well as model_info and model_results 
    dataframes that will be used for storing information about the models. 
    
    It then performs the operations necessary for creating, fitting, and making predictions with various random forest models, using 
    each of our list of feature combinations and varied values for the max-depth and min-samples-leaf hyperparameters. Information 
    about these models and their performance metrics are stored in the appropriate dataframes. 
    
    model_number, model_info, and model_results variables are returned (in that order). 
    '''

    # iterate over each feature set
    for features in feature_combos:
        # iterate over integers 1-10 to adjust max-depth value
        for max_depth in range(1, 11):
            # iterate over integers 1-10 to adjust min-samples-leaf value
            for min_samples_leaf in range(1, 11):
                
                # create a new model number by adding 1 to the previous model number
                model_number += 1
                
                # identify the model type
                model_type = 'random forest'

                # store info about the model
                
                # create a dictionary with the list of features and the hyperparameter values
                dct = {'model_number': model_number,
                    'model_type': model_type,
                    'features': features,
                    'max_depth': max_depth, 
                    'min_samples_leaf': min_samples_leaf}
                # append that dictionary to the model_info dataframe
                model_info = model_info.append(dct, ignore_index=True)

                # separate each sample into x (features) and y (target)
                x_train = train[features]
                y_train = train[target]

                x_validate = validate[features]
                y_validate = validate[target]


                # create the classifier
                
                # a random forest classifier with the given max-depth and min-samples-leaf values
                # establish a random state for produceability
                clf = RandomForestClassifier(max_depth=max_depth, 
                                            min_samples_leaf=min_samples_leaf, 
                                            random_state=42)

                # fit the classifier to the training data
                clf = clf.fit(x_train, y_train)


                # create predictions for the model's performance on the training set
                y_pred = clf.predict(x_train)


                # get metrics
                
                # create a dictionary for each performance metric type on the train sample
                #     append that dictionary to the model_results dataframe
                dct = {'model_number': model_number, 
                    'sample_type': 'train', 
                    'metric_type': 'accuracy',
                    'score': sk.metrics.accuracy_score(y_train, y_pred)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                    'sample_type': 'train', 
                    'metric_type': 'precision',
                    'score': sk.metrics.precision_score(y_train, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                    'sample_type': 'train', 
                    'metric_type': 'recall',
                    'score': sk.metrics.recall_score(y_train, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                    'sample_type': 'train', 
                    'metric_type': 'f1_score',
                    'score': sk.metrics.f1_score(y_train, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)


                # create prediction results for the model's performance on the validate sample
                y_pred = clf.predict(x_validate)

                # get metrics
                
                # create a dictionary for each performance metric type on the validate sample and 
                #    append that dictionary to the model_results dataframe
                dct = {'model_number': model_number, 
                    'sample_type': 'validate', 
                    'metric_type': 'f1_score',
                    'score': sk.metrics.f1_score(y_validate, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                    'sample_type': 'validate', 
                    'metric_type': 'accuracy',
                    'score': sk.metrics.accuracy_score(y_validate, y_pred)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                    'sample_type': 'validate', 
                    'metric_type': 'precision',
                    'score': sk.metrics.precision_score(y_validate, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)

                dct = {'model_number': model_number, 
                    'sample_type': 'validate', 
                    'metric_type': 'recall',
                    'score': sk.metrics.recall_score(y_validate, y_pred, pos_label=positive)}
                model_results = model_results.append(dct, ignore_index=True)    

    return model_number, model_info, model_results


def run_knn(train,
            validate,
            target,
            positive,
            feature_combos,
            model_number,
            model_info,
            model_results):
    '''
    This function takes in the train and validate samples as dataframes, the target variable label, the positive condition label,
    a list of feature combinations to be tested, an initialized model_number variable, as well as model_info and model_results 
    dataframes that will be used for storing information about the models. 
    
    It then performs the operations necessary for creating, fitting, and making predictions with various k-nearest-neighbor models, using 
    each of our list of feature combinations and varied values for the n_neighbors hyperparameter. Information 
    about these models and their performance metrics are stored in the appropriate dataframes. 
    
    model_number, model_info, and model_results variables are returned (in that order). 
    '''

    # iterate over each feature set
    for features in feature_combos:
        # iterate over integers 1-10 to change the K hyperparameter
        for k_neighbors in range(1, 11):
            
            # establish a new model number by adding one to the previous model number
            model_number += 1
            
            # identify the model type
            model_type = 'KNN'

            # store info about the model
            
            # create a dictionary containing model type, feature set, and hyperparameter values
            dct = {'model_number': model_number,
                'model_type': model_type,
                'features': features,
                'k-neighbors': k_neighbors}
            #    append that dictionary to the model_info dataframe
            model_info = model_info.append(dct, ignore_index=True)

            # separate each sample into x (features) and y (target)
            
            x_train = train[features]
            y_train = train[target]

            x_validate = validate[features]
            y_validate = validate[target]


            # create a K-nearest neighbors classifer with the given value for K
            clf = KNeighborsClassifier(n_neighbors=k_neighbors)

            # fit the classifier to the training data
            clf = clf.fit(x_train, y_train)

            # create prediction results for the model's performance on the training set
            y_pred = clf.predict(x_train)


            # get metrics
            
            # create a dictionary for each metric of performance on the training set, append that dictionary 
            #     model_results dataframe
            dct = {'model_number': model_number, 
                'sample_type': 'train', 
                'metric_type': 'accuracy',
                'score': sk.metrics.accuracy_score(y_train, y_pred)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                'sample_type': 'train', 
                'metric_type': 'precision',
                'score': sk.metrics.precision_score(y_train, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                'sample_type': 'train', 
                'metric_type': 'recall',
                'score': sk.metrics.recall_score(y_train, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                'sample_type': 'train', 
                'metric_type': 'f1_score',
                'score': sk.metrics.f1_score(y_train, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)


            # create prediction results for the model's performance on the validate sample
            y_pred = clf.predict(x_validate)

            # get metrics
            
            # create a dictionary for each metric of performance on the validate sample
            #     append that dictionary to the model-results dataframe
            dct = {'model_number': model_number, 
                'sample_type': 'validate', 
                'metric_type': 'f1_score',
                'score': sk.metrics.f1_score(y_validate, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                'sample_type': 'validate', 
                'metric_type': 'accuracy',
                'score': sk.metrics.accuracy_score(y_validate, y_pred)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                'sample_type': 'validate', 
                'metric_type': 'precision',
                'score': sk.metrics.precision_score(y_validate, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                'sample_type': 'validate', 
                'metric_type': 'recall',
                'score': sk.metrics.recall_score(y_validate, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)    

    return model_number, model_info, model_results


def run_logistic_regression(train,
                            validate,
                            target,
                            positive,
                            feature_combos,
                            model_number,
                            model_info,
                            model_results):
    '''
    This function takes in the train and validate samples as dataframes, the target variable label, the positive condition label,
    a list of feature combinations to be tested, an initialized model_number variable, as well as model_info and model_results 
    dataframes that will be used for storing information about the models. 
    
    It then performs the operations necessary for creating, fitting, and making predictions with various k-nearest-neighbor models, using 
    each of our list of feature combinations and varied values for the n_neighbors hyperparameter. Information 
    about these models and their performance metrics are stored in the appropriate dataframes. 
    
    model_number, model_info, and model_results variables are returned (in that order). 
    '''

    # iterate over each feature set
    for features in feature_combos:  
        # iterate over the given list of values for the C hyperparameter
        for c_value in [.001, .01, .1, 1, 10, 100, 1000]:
            
            # create a new model number by adding one to the previous model number
            model_number += 1
            
            # identify the model type
            model_type = 'logistic regression'

            # store info about the model
            
            # create a dictionary with info about the model, features, and hyperparameter values
            dct = {'model_number': model_number,
                'model_type': model_type,
                'features': features,
                'c_value': c_value}
            # append that dictionary to the model_info dataframe
            model_info = model_info.append(dct, ignore_index=True)

            # separate each sample into x and y
            x_train = train[features]
            y_train = train[target]

            x_validate = validate[features]
            y_validate = validate[target]


            # create a Logistic Regression classifier with the given value of C hyperparameter
            clf = LogisticRegression(C=c_value)

            # fit the classifier to the training data
            clf = clf.fit(x_train, y_train)

            # create prediction results for the model's performance on the training sample
            y_pred = clf.predict(x_train)

            # get metrics
            
            # create a dictionary for each metric of performance on the training sample
            #    append that dictionary to the model_results dataframe
            dct = {'model_number': model_number, 
                'sample_type': 'train', 
                'metric_type': 'accuracy',
                'score': sk.metrics.accuracy_score(y_train, y_pred)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                'sample_type': 'train', 
                'metric_type': 'precision',
                'score': sk.metrics.precision_score(y_train, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                'sample_type': 'train', 
                'metric_type': 'recall',
                'score': sk.metrics.recall_score(y_train, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                'sample_type': 'train', 
                'metric_type': 'f1_score',
                'score': sk.metrics.f1_score(y_train, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)


            # create prediction results for the validate sample
            y_pred = clf.predict(x_validate)

            # get metrics
            
            # create a dictionary for each metric of performance on the validate sample
            #    append that dictionary to the model_results dataframe
            dct = {'model_number': model_number, 
                'sample_type': 'validate', 
                'metric_type': 'f1_score',
                'score': sk.metrics.f1_score(y_validate, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                'sample_type': 'validate', 
                'metric_type': 'accuracy',
                'score': sk.metrics.accuracy_score(y_validate, y_pred)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                'sample_type': 'validate', 
                'metric_type': 'precision',
                'score': sk.metrics.precision_score(y_validate, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)

            dct = {'model_number': model_number, 
                'sample_type': 'validate', 
                'metric_type': 'recall',
                'score': sk.metrics.recall_score(y_validate, y_pred, pos_label=positive)}
            model_results = model_results.append(dct, ignore_index=True)    

    return model_number, model_info, model_results

def final_test(train, test, target, features):
    '''
    This function takes in the train and test samples, then fits a model using the train sample. This model 
    matches the one that was found to be our best performing model in our previous testing and evaluation.

    (Random Forest, all available features, max-depth=9, min-samples-leaf=7). 

    It then creates predictions for the test sample using this model, and print's the model's accuracy score.

    It returns the predictions as y_pred and the probability of those predictions as y_pred_proba (in that order)
    '''
    # re-creating the model using the given features and hyperparameters

    # separate each sample into x (features) and y (target)
    x_train = train[features]
    y_train = train[target]

    x_test = test[features]
    y_test = test[target]

    # create the classifier

    # a random forest classifier with the given max-depth and min-samples-leaf values
    # establish a random state for produceability
    clf = RandomForestClassifier(max_depth=9, 
                                    min_samples_leaf=7, 
                                    random_state=42)

    # fit the classifier to the training data
    clf = clf.fit(x_train, y_train)

    # create predictions for the model's performance on the test set
    y_pred = clf.predict(x_test)

    # establish the probability for those predictions created above
    y_pred_proba = clf.predict_proba(x_test)[:,1]

    print(f'Model Accuracy: {sk.metrics.accuracy_score(y_test, y_pred):.1%}')

    return y_pred, y_pred_proba


def create_predictions(test, y_pred, y_pred_proba):
    '''
    This function takes in the test sample, and the previously created predictions and probabilities 
    of those predictions, then writes the predictions to a local csv file, along with the corresponding 
    customer_id's. 
    '''
    # create a dataframe with the appropriate columns
    predictions = pd.DataFrame(columns=['customer_id', 'churn_probability', 'churn_prediction'])
    # set customer id using the customer_id column from the original database
    predictions['customer_id'] = test.customer_id
    # set the churn_probability column using the probabilities created above
    predictions['churn_probability'] = y_pred_proba
    # set the predictions column using the predictions created above
    predictions['churn_prediction'] = y_pred
    # write to a local csv file
    predictions.to_csv('predictions.csv')