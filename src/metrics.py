import logging

import pandas as pd
from fuzzywuzzy import fuzz
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def levenshtein_distance_sort(row, it_name) ->  int:
    return fuzz.ratio(it_name, row['name'])


def calculate_skill_similarity(row, it_skills) -> tuple[int, int]:
    gb_skills = row['stack']
    
    # Объединяем навыки в одну строку
    gb_skills_str = ' '.join(gb_skills)
    it_skills_str = ' '.join(it_skills)
    
    # Расстояние Левенштейна между строками навыков
    levenshtein_distance = fuzz.ratio(gb_skills_str, it_skills_str)
    
    # Максимальное пересечение множеств между навыками
    gb_skills_counter = Counter(gb_skills)
    it_skills_counter = Counter(it_skills)
    intersection = gb_skills_counter & it_skills_counter
    max_intersection = sum(intersection.values())
    
    return levenshtein_distance, max_intersection


def calculate_tfidf_similarity(data_gb, data_it, vacancy) -> pd.DataFrame:
    # Объединяем все текстовые данные из data_gb в один список
    vacancy_texts = [vacancy['name'] + vacancy['description'] + ' '.join(vacancy['key_skills'])]

    gb_texts = list(data_it['description'] + [' '.join(i) for i in data_it['hard_skills']]) + vacancy_texts + list(data_gb['summary'] + data_gb['other']+ [' '.join(i) for i in data_gb['stack']])
    # Создаем TF-IDF векторизатор
    tfidf_vectorizer_gb = TfidfVectorizer()

    # Применяем TF-IDF векторизацию к текстовым данным из data_gb
    tfidf_matrix = tfidf_vectorizer_gb.fit_transform(gb_texts)

    # Преобразуем описание из data_it в TF-IDF вектор
    tfidf_matrix_it = tfidf_matrix[:len(data_it)+1]
    tfidf_matrix_gb = tfidf_matrix[len(data_it)+1:]

    # Вычисляем косинусное сходство между описанием из data_it и каждым элементом из data_gb
    cosine_similarities = cosine_similarity(tfidf_matrix_it[-1], tfidf_matrix_gb)

    # Добавляем рейтинг сходства в data_gb
    data_gb['tfidf_description'] = cosine_similarities[0]

    return data_gb