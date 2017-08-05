import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics
import sklearn.model_selection

def train_ensemble(ensemble, X_train, y_train):
    for output_idx, model in enumerate(ensemble):
        model.fit(X_train, y_train.iloc[:, output_idx])

def score_ensemble(ensemble, X_test, y_test):

    score = np.zeros(len(ensemble))
    y_pred = np.zeros(y_test.shape)

    for output_idx, model in enumerate(ensemble):
        y = y_test.iloc[:, output_idx]
        y_hat = model.predict(X_test)
        y_pred[:, output_idx] = y_hat

    score = sklearn.metrics.r2_score(y_test, y_pred, multioutput='variance_weighted')

    return score

def MCRMSE(model, X, y):

    if type(model) is list:
        y_pred = np.zeros(y.shape)
        for output_idx,sub_model in enumerate(model):
            y_pred[:, output_idx] = sub_model.predict(X)
    else:
        y_pred = model.predict(X)

    error = y_pred - y
    root_mean_square_error = np.sqrt(np.mean(np.square(error),axis=0))
    mean_columnwise_root_mean_square_error = np.mean(root_mean_square_error)

    return mean_columnwise_root_mean_square_error

def compare_models(models, model_names, PCA, X, y, num_splits=20):
    num_models = len(models)

    train_scores = np.zeros((num_splits, num_models))
    test_scores = np.zeros((num_splits, num_models))

    for model_idx, (model, name) in enumerate(zip(models,model_names)):
        for split_idx in range(num_splits):

            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                       test_size=0.2)

            X_train = PCA.fit_transform(X_train)
            X_test = PCA.transform(X_test)

            if type(model) != list:
                model.fit(X_train, y_train)
                #train_score = model.score(X_train, y_train)
                #test_score = model.score(X_test, y_test)
            else:
                train_ensemble(model, X_train, y_train)
                #train_score = scoreSVR(model, X_train, y_train)
                #test_score =scoreSVR(model, X_test, y_test)

            train_score = MCRMSE(model, X_train, y_train)
            test_score = MCRMSE(model, X_test, y_test)
            train_scores[split_idx, model_idx] = train_score
            test_scores[split_idx, model_idx] = test_score

    return train_scores, test_scores

def comparison_plot(model_names, train_scores, test_scores):
    num_splits, num_models = train_scores.shape
    mean_trains = np.mean(train_scores, axis = 0)
    std_trains = np.std(train_scores, axis = 0, ddof=1)
    mean_tests = np.mean(test_scores, axis = 0)
    std_tests = np.std(test_scores, axis = 0 , ddof=1)

    plt.figure(figsize=(12,6))
    width = 0.1

    xmin = 0.6-width*2; xmax = 0.6+width*(num_models+1.5)
    plt.hlines([0.4, 0.36], xmin, xmax, linewidth=4, label='Kaggle Top 10%',
                              linestyle='--', alpha=0.5, color='gray',
                                )

    for model_idx,name in enumerate(model_names):
        #ys = [mean_trains[model_idx], mean_tests[model_idx]]
        #xs = [0+width*model_idx, 0.6+width*model_idx]
        ys = [mean_tests[model_idx]]
        xs = [0.6+width*model_idx]
        plt.bar(xs, ys, width = width, label=name, alpha=0.75, linewidth=4, edgecolor='white')
        #plt.errorbar(xs, [mean_trains[model_idx], mean_tests[model_idx]],
        #             yerr = [std_trains[model_idx], std_tests[model_idx]],
        #            linewidth=4,
        #             color='grey', linestyle='None',
        #            )
        plt.errorbar(xs, ys,
             yerr = [std_tests[model_idx]],
            linewidth=4,
             color='grey', linestyle='None',
            )

    plt.xlim([xmin, xmax])
    plt.yticks([0,0.5,1.0])
    plt.xticks([0.6+width*(num_models//2)], ["Test"],)
    plt.ylabel('RMS Error')
    plt.title("Model Performance", fontweight='bold',)
    plt.legend(loc=(1.01,0.6));
