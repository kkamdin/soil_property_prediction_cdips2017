import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics
import sklearn.model_selection

def trainSVR(SVR_models, X_train, y_train):
    for output_idx, SVR_model in enumerate(SVR_models):
        SVR_model.fit(X_train, y_train.iloc[:, output_idx])

def scoreSVR(SVR_models, X_test, y_test):

    score = np.zeros(len(SVR_models))
    y_pred = np.zeros(y_test.shape)

    for output_idx,SVR_model in enumerate(SVR_models):
        y = y_test.iloc[:, output_idx]
        y_hat = SVR_model.predict(X_test)
        y_pred[:, output_idx] = y_hat

    score = sklearn.metrics.r2_score(y_test, y_pred, multioutput='variance_weighted')

    return score

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

            if name != 'Support Vector Machine':
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
            else:
                trainSVR(model, X_train, y_train)
                train_score = scoreSVR(model, X_train, y_train)
                test_score =scoreSVR(model, X_test, y_test)

            train_scores[split_idx, model_idx] = train_score
            test_scores[split_idx, model_idx] = test_score

    return train_scores, test_scores

def comparison_plot(model_names, train_scores, test_scores):
    num_splits, num_models = train_scores.shape
    mean_trains = np.mean(train_scores, axis = 0)
    sem_trains = np.std(train_scores, axis = 0, ddof=1)/np.sqrt(num_splits)
    mean_tests = np.mean(test_scores, axis = 0)
    sem_tests = np.std(test_scores, axis = 0 , ddof=1)/np.sqrt(num_splits)

    plt.figure(figsize=(12,6))
    width = 0.1

    for model_idx,name in enumerate(model_names):
        ys = [mean_trains[model_idx], mean_tests[model_idx]]
        xs = [0+width*model_idx, 0.6+width*model_idx]
        plt.bar(xs, ys, width = width, label=name, alpha=0.75, linewidth=4, edgecolor='white')
        plt.errorbar(xs, [mean_trains[model_idx], mean_tests[model_idx]],
                     yerr = [sem_trains[model_idx], sem_tests[model_idx]],
                    linewidth=4,
                     color='grey', linestyle='None',
                    )

    plt.xlim([0-width*2, 0.6+width*(num_models+1.5)])
    plt.yticks([0,0.5,1.0])
    plt.xticks([0+width*1.5,0.6+width*1.5], ["Train", "Test"], fontsize='xx-large')
    plt.ylabel(r'$R^2$')
    plt.title("Model Performance", fontweight='bold',)
    plt.legend(loc=(1.01,0.6));
