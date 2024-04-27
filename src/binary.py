from joblib import dump, load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from utils import preprocess_text


def train_binary_trebovanyia_model(data_csv_path: str) -> None:
    
    data = pd.read_csv(data_csv_path)
    
    # Предобработка текста
    data['preprocessed_text'] = data['text'].apply(preprocess_text)

    # Разделение данных на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(data['preprocessed_text'], data['label'], test_size=0.2, random_state=42, shuffle=True)

    # Преобразование текста в векторы TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Создание и обучение модели классификации (например, Multinomial Naive Bayes)
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)

    # Предсказание на тестовом наборе
    y_pred = clf.predict(X_test_tfidf)

    # Оценка точности модели
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Отчет о классификации
    print(classification_report(y_test, y_pred))

    dump(clf, 'clf.joblib')


def load_trained_clf(clf_path: str) -> MultinomialNB:

    clf = load(clf_path)

    return clf
