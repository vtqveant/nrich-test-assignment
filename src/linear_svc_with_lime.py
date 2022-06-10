import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import metrics
from lime.lime_text import LimeTextExplainer

categories = [
    '11-1021.00',
    '11-2021.00',
    '11-2022.00',
    '11-3031.02',
    '13-1111.00',
    '13-2051.00',
    '15-1121.00',
    '15-1122.00',
    '15-1132.00',
    '15-1133.00',
    '15-1134.00',
    '15-1142.00',
    '15-1151.00',
    '29-1141.00',
    '31-1014.00',
    '33-3021.06',
    '41-2031.00',
    '43-4051.00',
    '49-3023.02',
    '49-9071.00',
    '53-3032.00'
]

data_train = pd.read_csv('../data/train_df.csv')
data_test = pd.read_csv('../data/test_df.csv')

X_train, X_test = data_train.Title, data_test.Title
y_train, y_test = data_train.Code, data_test.Code

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

y_pred = pipeline.predict(X_test)
score = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(y_test, y_pred, target_names=categories))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))


# Explaining individual predictions
clf_predictions_df = pd.DataFrame(
    data={
        "text": data_test.Title,
        "real category": data_test.Code,
        "predicted category": y_pred
    }
)

clf_predictions_df["text length"] = clf_predictions_df["text"].str.len()
clf_predictions_df.sort_values(by="text length", ascending=False, inplace=True)

# samples which has been correctly classified
correct_classified_df = clf_predictions_df[
    clf_predictions_df["real category"] == clf_predictions_df["predicted category"]
    ]

# samples which has been incorrectly classified
incorrect_classified_df = clf_predictions_df[
    clf_predictions_df["real category"] != clf_predictions_df["predicted category"]
    ]

# preprocessing function used in the pipeline to analyze input documents
tokenize_fn = pipeline.named_steps['vectorizer'].build_analyzer()
explainer = LimeTextExplainer(verbose=True, class_names=pipeline.classes_)
# classifier = pipeline.named_steps['classifier']
clf_fn = pipeline.predict_proba

# get an explanation
correct_sample = correct_classified_df.loc[[10]]
text_to_explain = correct_sample["text"].values[0]
cleaned_text_to_explain = " ".join(tokenize_fn(text_to_explain))

exp_object = explainer.explain_instance(
    text_instance=cleaned_text_to_explain,
    classifier_fn=clf_fn,
    top_labels=1,
    num_features=10
)
labels = exp_object.available_labels()
print(labels)

# exp_object.save_to_file(
#     file_path="../target/correct-classification-explanation.html",
#     labels=labels,
# )

print()
for label in labels:
    lst = exp_object.as_list(label=label)
    print(lst)
