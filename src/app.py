import streamlit as st

import warnings
warnings.filterwarnings('ignore')

from utils import preprocess_and_match_vacancy


def main():
    st.title("Поиск подходящих курсов от GeekBrains")

    vacancy_url = st.text_input("Введите ссылку на вакансию:")

    # Кнопка для поиска схожего курса GeekBrains
    if st.button("Найти схожий курс от GeekBrains"):
        # Преобразуем текст в верхний регистр и выводим его
        data, vacancy = preprocess_and_match_vacancy(vacancy_url=vacancy_url, data_gb_path='data\data_gb_resultv1.csv', data_it_path='data\preprocess_it.csv')
        st.write("Описание схожего курса GeekBrains:", data.loc[0]['summary'])
        st.write("Описание вакансии:", vacancy['description'])


if __name__ == "__main__":
    main()
