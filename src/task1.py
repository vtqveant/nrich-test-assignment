import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import metrics
from lime.lime_text import LimeTextExplainer


def compute_metrics(pipeline, X_test, y_test, target_names):
    """
    Explaining individual predictions using LIME
    """
    y_pred = pipeline.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    print("accuracy:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print()


def explain_prediction(pipeline, text):
    if len(text) == 0 or text is None:
        raise AssertionError('Invalid text')

    [y_pred] = pipeline.predict([text])
    print("Predicted category: {}\n".format(y_pred))

    explainer = LimeTextExplainer(verbose=False, class_names=pipeline.classes_)
    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=pipeline.predict_proba,
        top_labels=1,
        num_features=10
    )

    data = explanation.as_list(label=explanation.available_labels()[0])
    df = pd.DataFrame(data, columns=['token', 'score'])
    top_5 = df.sort_values(by=['score'], ascending=False).head(5)
    print(top_5)


def main():
    categories = [
        '11-1021.00', '11-2021.00', '11-2022.00', '11-3031.02', '13-1111.00', '13-2051.00', '15-1121.00', '15-1122.00',
        '15-1132.00', '15-1133.00', '15-1134.00', '15-1142.00', '15-1151.00', '29-1141.00', '31-1014.00', '33-3021.06',
        '41-2031.00', '43-4051.00', '49-3023.02', '49-9071.00', '53-3032.00'
    ]

    data_train = pd.read_csv('../data/train_df.csv')
    data_test = pd.read_csv('../data/test_df.csv')

    X_train, X_test = data_train.Title, data_test.Title
    y_train, y_test = data_train.Code, data_test.Code

    # A model and a pipeline
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english")

    # LinearSVC with L1-based feature selection;
    # The smaller C, the stronger the regularization. The more regularization, the more sparsity.
    svm = LinearSVC(penalty="l2")

    # a decorator to have predict_proba to be used by LIME
    svm = CalibratedClassifierCV(base_estimator=svm, cv=5, method="isotonic")

    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("feature_selection", SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
        ("classifier", svm)
    ])

    pipeline.fit(X_train, y_train)

    compute_metrics(pipeline, X_test, y_test, categories)
    explain_prediction(pipeline, 'ecm project manager oconus position')


if __name__ == '__main__':
    main()
