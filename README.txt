
  Задача 1
--------------

Решалась стандартным средствами scikit-learn, подбор параметров модели и т.п. не производился.
Сначала была выполнена грубая оценка призводительности различных классификаторов (см. task1_model_screening.ipynb,
результаты: accuracy в пределах 0.37 - 0.46 по 15 различным классификаторам из scikit-learn).
Я пришел к выводу, что без доп. работы с фичами и fine-tuning модели результаты по всем классификаторам сопоставимы
и остановился на LinearSVC, т.к. он давал чуть лучшее значение accuracy, приняв его за baseline.
Интерепретация результатов делалась с помощью LIME.


  Задача 2
-------------

Я остановился на оптимизации LinearSVC главным образом по причине скорости обучения и т.к. посчитал, что её
перформанса будет достаточно для демонстрации подхода в целом.

На основе EDA были сделаны следующие наблюдения:
    * множество классов разбивается примерно на четыре группы, которые оказываются достаточно сбалансированными
    * тексты подвергались предобработке и содержат большое число артефактов препроцессора (подробнее см. eda.ipynb)

В связи с этим я посчитал нецелесообразным привлечение сложных методов для работы с текстом и решил
ограничиться использованием n-грамм на уровне слов и отдельных символов.

Дополнительно предпринята попытка расшифровки аббревиатур с указанием группы (Medical, Manager, IT, Analyst).
В данном случае, для проверки концепции, это было реализовано с помощью данных, собранных вручную [1]. В промышленном
решении коллекция аббревиатур может быть собрана из источников вроде DBPedia с последующей ручной фильтрацией,
а признак группы может быть проставлен с помощью дополнительного классификатора. Сравнение результатов моделей
с использованием расшифровки аббервиатур и без неё (выбор модели выполнялся с помощью grid search, сравнение моделей
по f1 и accuracy на тестовой выборке) показало значительное уменьшение влияния символьных n-грамм, что согласуется с
ожиданием.

Лучшая модель показала accuracy 0.472 на 5-fold cross-validation на обучающей выборке и 0.486 на тестовой выборке
(улучшение на 4.97% или 0.023 в абсолютных значениях по сравнению с baseline).



  Ссылки
------------

  [1] https://www.asha.org/siteassets/practice-portal/medicalabbreviations.pdf