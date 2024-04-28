import logging
from typing import Any
import re
import string

import numpy as np
import pandas as pd
from parse_hh_data import download
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

from configuration import PATTERN
from metrics import levenshtein_distance_sort, calculate_skill_similarity, calculate_tfidf_similarity
from sklearn.preprocessing import MinMaxScaler
from joblib import load

STOP_WORDS = set(stopwords.words('russian'))
PUNCTUATION = set(string.punctuation)

binar_model = load('clf.joblib')
tfidf_vectorizer = load('tfidf_vectorizer.joblib')


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_data_from_vacancy_url(vacancy_url: str) -> dict[str, Any]:
    """_summary_

    Args:
        vacancy_url (str): _description_

    Returns:
        dict[str, Any]: _description_
    """
    # Get vacancy data from url
    vacancy_id = re.sub(PATTERN, r'\1', vacancy_url)
    vacancy = download.vacancy(vacancy_id)

    # Parsing keys
    vacancy['key_skills'] = [i['name'] for i in vacancy['key_skills']]
    keys_to_keep = ['name', 'description', 'key_skills', 'salary', 'experience']

    # Create a new dictionary with only the specified keys
    vacancy = {key: vacancy[key] for key in keys_to_keep if key in vacancy}

    # Parse HTML
    soup = BeautifulSoup(vacancy['description'], 'html.parser')

    # Extract text
    text_split = soup.get_text()
    text_split = re.split(r'\s{2,}|;|[.!?]', text_split)
    parsed_description = [i.lower().strip() for i in text_split if i != '']

    # Parsing data
    vacancy['description'] = '\n'.join(parsed_description)
    vacancy['sentences'] = parsed_description
    vacancy['name'] = vacancy['name'].lower()
    vacancy['key_skills'] = [i.lower() for i in vacancy['key_skills']]

    return vacancy



def predict_sentences(sentences):
    # Make predictions for each sentence
    sentences = tfidf_vectorizer.transform(sentences)
    predictions = binar_model.predict(sentences)

    return predictions


def get_requirements_predict(preprocessed_vacancy):
    sentences = preprocessed_vacancy['sentences']

    # Get predictions for the sentences
    predictions = predict_sentences(sentences)
    replacement_dict = {0: False, 1: True}

    # Замена значений в массиве с помощью словаря
    arr_replaced = np.array([replacement_dict[val] for val in predictions])

    preprocessed_vacancy['requirements_predict'] = np.array(sentences)[arr_replaced]

    return preprocessed_vacancy


def preprocess_text(text: str) -> str:
    """_summary_

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in STOP_WORDS and word not in PUNCTUATION]
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens).lower()
    return preprocessed_text


def preprocess_skills(skills: list[str]) -> str:
    """_summary_

    Args:
        skills (list[str]): _description_

    Returns:
        str: _description_
    """
    return ';'.join(skill.strip().lower() for skill in skills[1:-1].split(','))

def postprocessing_vacancy_data(vacancy: dict[str, Any]) -> dict[str, Any]:
    vacancy['description'] = preprocess_text(vacancy['description'])
    vacancy['name'] = vacancy['name'].lower()
    return vacancy


def preprocess_gb_data(csv_file_path: str) -> pd.DataFrame:
    """_summary_

    Args:
        csv_file_path (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Load DataFrame
    data = pd.read_csv(csv_file_path)

    # Rename column for convenience
    data['stack'] = data['Технологии, инструменты']

    # Select only required columns
    data = data[['index', 'name', 'url', 'stack', 'summary', 'other']]

    # Preprocess 'stack' column
    data['stack'] = data['stack'].str.lower().str.split(',')


    # Preprocess 'summary' column
    data['summary'] = data['summary'].apply(preprocess_text)

    # Preprocess 'other' column
    data['other'] = data['other'].apply(preprocess_text)

    data['name'] = data['name'].apply(lambda x: x.lower())

    return data


def preprocess_it_csv(csv_file_path: str) -> pd.DataFrame:
    # Load DataFrame
    data = pd.read_csv(csv_file_path, sep=';')

    # Remove rows with NaN values in 'description' and 'name' columns
    data = data[~data['description'].isna()].reset_index(drop=True)
    data = data[~data['name'].isna()].reset_index(drop=True)

    # Select only required columns
    data = data[['name', 'description', 'hard_skills']]

    # Preprocess 'hard_skills' column
    data['hard_skills'] = data['hard_skills'].apply(preprocess_skills)

    # Preprocess 'description' column
    data['description'] = data['description'].apply(preprocess_text)

    # Lowercase 'name' column
    data['name'] = data['name'].apply(lambda x: x.lower())

    return data

def preprocess_and_match_vacancy(vacancy_url: str, data_gb_path: str, data_it_path: str):
    # Preprocess vacancy from url
    logging.info("Start preprocess_data_from_vacancy_url")
    preprocessed_vacancy = preprocess_data_from_vacancy_url(vacancy_url)
    logging.info("End preprocess_data_from_vacancy_url")

    # Preprocess vacancy data
    logging.info("Start postprocessing_vacancy_data")
    preprocessed_vacancy = postprocessing_vacancy_data(preprocessed_vacancy)
    logging.info("End postprocessing_vacancy_data")

    logging.info("Start preprocess_gb_data")
    data_gb = preprocess_gb_data(data_gb_path)
    logging.info("End preprocess_gb_data")

    logging.info("Start preprocess_it_csv")
    # data_it = preprocess_it_csv(data_it_path)
    data_it = pd.read_csv(data_it_path, index_col=0)
    data_it = data_it.replace(np.nan, '', regex=True)
    logging.info("End preprocess_it_csv")

    logging.info("Start calculate Levenshtein")
    # Calculate Levenshtein distance between vacancy name and company names in data_gb
    data_gb['levenshtein_distance'] = data_gb.apply(levenshtein_distance_sort, args=(preprocessed_vacancy['name'],), axis=1)

    # Extract IT skills from vacancy
    it_skills = preprocessed_vacancy['key_skills']

    # Calculate skill similarity between vacancy and job descriptions in data_gb
    data_gb['levenshtein_distance_stack'], data_gb['max_intersection_stack'] = zip(*data_gb.apply(calculate_skill_similarity, args=(it_skills,), axis=1))
    logging.info("End calculate Levenshtein")

    logging.info("Start calculate_tfidf_similarity")
    # Calculate TF-IDF similarity between vacancy description and job descriptions in data_gb
    data_gb = calculate_tfidf_similarity(data_gb, data_it, preprocessed_vacancy)
    logging.info("Start calculate_tfidf_similarity")

    logging.info("Start scaling")
    # Нормализация метрик
    scaler = MinMaxScaler()
    metrics_to_normalize = ['levenshtein_distance', 'levenshtein_distance_stack', 'max_intersection_stack', 'tfidf_description']
    data_gb['total_score'] = scaler.fit_transform(data_gb[metrics_to_normalize]).sum(axis=1)
    data_gb = data_gb.sort_values(by='total_score', ascending=False).reset_index(drop=True)
    logging.info("End scaling")

    return data_gb, preprocessed_vacancy


def get_matrix_df(vacancy_url: str, data_gb_path: str):

    logging.info("Start preprocess_data_from_vacancy_url")
    preprocessed_vacancy = preprocess_data_from_vacancy_url(vacancy_url)
    logging.info("End preprocess_data_from_vacancy_url")

    preprocessed_vacancy = get_requirements_predict(preprocessed_vacancy)

    logging.info("Start preprocess_gb_data")
    data_gb = preprocess_gb_data(data_gb_path)
    logging.info("End preprocess_gb_data")
    # Создание пустого DataFrame для матрицы
    # Создание пустого DataFrame для матрицы
    matrix_df = pd.DataFrame(index=preprocessed_vacancy['requirements_predict'], columns=data_gb.index)

    # Установка параметров для окна
    # Длина окна, которая должна быть примерно равной длине требования
    threshold = 65  # Порог сходства
    # print(preprocessed_vacancy['requirements_predict'])
    # Проходимся по каждому требованию и каждому элементу столбца 'other' и заполняем матрицу
    for req in preprocessed_vacancy['requirements_predict'][:10]:
        window_size = len(req)
        for idx, other_text in data_gb['other'].items():
            max_similarity = 0
            for i in range(0, min(4000, len(other_text)), 4):
                window = other_text[i:i+window_size].lower()
                similarity = fuzz.partial_ratio(req.lower(), window)
                max_similarity = max(max_similarity, similarity)
            if max_similarity >= threshold:
                matrix_df.at[req, idx] = True
            else:
                matrix_df.at[req, idx] = False

    # Заполнение пропущенных значений False
    matrix_df = matrix_df.fillna(False)

    return matrix_df
