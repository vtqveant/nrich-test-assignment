import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn import metrics

from pprint import pprint
from time import time


def compute_metrics(pipeline, X_test, y_test, target_names):
    y_pred = pipeline.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    print("accuracy:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print()


class AcronymExpander:
    """
    Just as a proof of concept, attempt to expand acronyms with a manually curated list.
    """

    def __init__(self, filename):
        d = pd.read_csv(filename, index_col='acronym').squeeze('columns').to_dict()
        self.expansions = d['expansion']
        self.subject = d['subject']

    def expand(self, text):
        exp = list()  # to preserve word order of expansions
        subj = set()  # to avoid duplicates
        for token in text.split(' '):
            if token in self.expansions.keys():
                exp.append(self.expansions[token])
                subj.add(self.subject[token])
        return text + ' ' + ' '.join(exp) + ' ' + ' '.join(list(subj))


def main():
    categories = [
        '11-1021.00', '11-2021.00', '11-2022.00', '11-3031.02', '13-1111.00', '13-2051.00', '15-1121.00', '15-1122.00',
        '15-1132.00', '15-1133.00', '15-1134.00', '15-1142.00', '15-1151.00', '29-1141.00', '31-1014.00', '33-3021.06',
        '41-2031.00', '43-4051.00', '49-3023.02', '49-9071.00', '53-3032.00'
    ]

    data_train = pd.read_csv('../data/train_df.csv')
    data_test = pd.read_csv('../data/test_df.csv')

    acronym_expander = AcronymExpander(filename='../resources/acronyms.csv')
    data_train['Title'] = data_train['Title'].apply(acronym_expander.expand)
    data_test['Title'] = data_test['Title'].apply(acronym_expander.expand)

    # for convenience
    X_train, X_test = data_train.Title, data_test.Title
    y_train, y_test = data_train.Code, data_test.Code

    # A model and a pipeline
    features = FeatureUnion(transformer_list=[
        ("char_ngram", TfidfVectorizer(analyzer='char')),  # to capture abbreviations and typos
        ("word_ngram", TfidfVectorizer(analyzer='word')),  # to capture tokens and MWEs
    ], n_jobs=-1)

    svm = LinearSVC(penalty="l2")
    clf = CalibratedClassifierCV(base_estimator=svm, cv=5, method="isotonic")

    pipeline = Pipeline([
        ("features", features),
        ("feature_selection", SelectFromModel(estimator=LinearSVC(penalty="l1", dual=False, tol=1e-3))),
        ("classifier", clf)
    ])

    # Parameters to use for grid search
    # Actually, this is the best variant, I omit the actual search as it takes a lot of time.
    parameters = {
        "features__char_ngram__max_df": [0.14],
        "features__char_ngram__sublinear_tf": [True],
        'features__char_ngram__ngram_range': [(1, 3)],
        "features__word_ngram__max_df": [0.42],
        "features__word_ngram__sublinear_tf": [True],
        'features__word_ngram__ngram_range': [(1, 2)],
        # 'features__word_ngram__max_features': [10000, 20000],
        'features__transformer_weights': [
            {"char_ngram": 0.5, "word_ngram": 0.5},
        ],
        # SVC supports class weighting for unbalanced classification
        'classifier__base_estimator__class_weight': [
            # {'29-1141.00': 0.3, '15-1142.00': 0.3, '31-1014.00': 3.0, '15-1121.00': 3.0},
            {}
        ]
    }

    # grid search
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # evaluate the best variant on a test set
    pipeline = grid_search.best_estimator_
    compute_metrics(pipeline, X_test, y_test, categories)


if __name__ == '__main__':
    main()
