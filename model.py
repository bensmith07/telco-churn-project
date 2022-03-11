# define a function for displaying model_results dataframes as a pivot table
#    for easier comparison of train vs validate performance
def display_model_results(model_results):
    return model_results.pivot_table(columns='model_number', 
                                     index=('metric_type', 'sample_type'), 
                                     values='score',
                                     aggfunc=lambda x: x)
                                     

def get_best_model_results(model_results, metric_type='accuracy', n_models=3):
    best_models = (model_results[(model_results.metric_type == metric_type) 
                               & (model_results.sample_type == 'validate')]
                                                 .sort_values(by='score', 
                                                              ascending=False)
                                                 .head(n_models)
                                                 .model_number
                                                 .values)

    best_model_results = model_results[(model_results.model_number == best_models[0]) 
                                     | (model_results.model_number == best_models[1]) 
                                     | (model_results.model_number == best_models[2])]

    return best_model_results