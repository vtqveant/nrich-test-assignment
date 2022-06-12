Также высылаю небольшое пояснение от команды:

В задании "The data"  во втором пункте под "обеспечить максимальную производительность" подразумевается,
что можно воспользоваться любыми NLP инструментами, даже если это повлияет на интерпретируемость.
Если модель не показала супер прироста производительности, это нормально, просто мы хотим увидеть: какие методы
вы выберете, как ими воспользуетесь, и что получите. Другими словами, нет необходимости в 100% точности,
мы хотим посмотреть на Ваш процесс решения задачи, даже если результат слегка лучше, чем в первом пункте.



Задача 1

Решалась стандартным средствами scikit-learn, подбор параметров модели и т.п. не производился.
Сначала была выполнена грубая оценка призводительности различных классификаторов (см. task1_model_screening.ipynb,
результаты: accuracy в пределах 0.37 - 0.46 по 15 различным классификаторам из scikit-learn).
Я пришел к выводу, что без доп. работы с фичами и fine-tuning модели результаты по всем классификаторам сопоставимы
и остановился на LinearSVC, т.к. он давал чуть лучшее значение accuracy, приняв его за baseline.
Интерепретация результатов делалась с помощью LIME.


Задача 2

Из 15 классификаторов из scikit-learn, рассмотренных в первой части, я выбрал те, что показали лучшие результаты, а
также добавил XGBoost. В итоге в части 2 я рассматриваю следующие модели:
    BernoulliNB
    MultinomialNB
    SGD Classifier
    LinearSVC
    Random Forest
    K-nn
    Ridge Classifier
    XGBoost

Выбор оптимальной модели делался по следующей схеме:

1. EDA для определения сбалансированности классов, состава словаря, возможности привлечения доп. информации
(word embeddings, knowledge bases, морфологии/стемминга и пр.)
2. Feature engineering (tf-idf, n-gramms, NER, w2v и пр.), feature bucketing, feature hashing, нормализация признаков
3. Оценка feature importance
4. Подбор гиперпараметров и схемы регуляризации для разных моделей с помощью grid search
5. Построение ансамбля
6. Оценка перформанса моделей по метрикам для
    а) классификации на сбалансированном датасете
    б) классификации на несбалансированном датасете


-------

clf = LinearSVC(penalty="l2")

Best score: 0.472
Best parameters set:
	features__char_ngram__max_df: 0.95
	features__char_ngram__ngram_range: (1, 4)
	features__transformer_weights: {'char_ngram': 0.5, 'word_ngram': 0.5}
	features__word_ngram__max_df: 0.65
	features__word_ngram__ngram_range: (1, 2)

accuracy on test set:   0.489


----

Best score: 0.473
Best parameters set:
	features__char_ngram__max_df: 0.14
	features__char_ngram__ngram_range: (1, 3)
	features__char_ngram__sublinear_tf: True
	features__transformer_weights: {'char_ngram': 0.5, 'word_ngram': 0.5}
	features__word_ngram__max_df: 0.42
	features__word_ngram__ngram_range: (1, 2)
	features__word_ngram__sublinear_tf: True

accuracy on training set: 0.484

----

https://www.asha.org/siteassets/practice-portal/medicalabbreviations.pdf