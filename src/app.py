import streamlit as st
import requests

import warnings
warnings.filterwarnings('ignore')

from utils import preprocess_and_match_vacancy, get_matrix_df
from pdf_parsing import ResumeSkillsFinder
from shpad.shpad import compare_models

def main():
    #st.header("ikanam-ai✔️", divider='rainbow')
    st.title("Поиск подходящих курсов от GeekBrains")
    txt = st.text_area(
        "⚙️ Как это работает?",
        "Модель анализирует информацию по вакансиям, а также по имеющимся "
        "образовательным программам. С помощью NLP-методов составляется рейтинг "
        "программ, а также составляется рекомендация к конкретной программе."
        )

    tab1, tab2, tab3 = st.tabs(["Вакансия😎📊", "Вакансия+резюме🚀💻", "Кластерный анализ🧠🤖"])
    

    with tab1:
        vacancy_url = st.text_input("Введите ссылку на вакансию:")

        # Кнопка для поиска схожего курса GeekBrains
        if st.button("Найти схожий курс от GeekBrains"):
            data, vacancy = preprocess_and_match_vacancy(vacancy_url=vacancy_url, data_gb_path='data/data_gb_resultv1.csv', data_it_path='data/preprocess_it.csv')
            # Выводим описание вакансии
            st.markdown(f"**Название вакансии:** {vacancy['name']}")

            # Проверка наличия ключевых навыков и их вывод с использованием Markdown
            if vacancy['key_skills']:
                st.markdown(f"**Стек:** {' , '.join(vacancy['key_skills'])}")
            st.write("---")

            st.subheader("Наиболее подходящие курсы для прохождения:")
            st.markdown(f"**{data.loc[0]['name'].upper()}**")
            score = data.loc[0]['total_score']
            percent = 100 if score <= 0 else round(100 * (score / (score + 1)), 2)
            st.markdown(
                f"<span style='color:green'>{'Схожесть по переплексии описания и навыков: '+str(percent)+'%'}</span>",
                unsafe_allow_html=True
            )
            st.markdown(f'<a href="{"https://raw.githubusercontent.com/ikanam-ai/VacancySkillsEstimator/main/logo.jpg"}"><img src="{"https://raw.githubusercontent.com/ikanam-ai/VacancySkillsEstimator/main/logo.jpg"}" alt="Foo" width="500" height="250"/></a>', unsafe_allow_html=True)
            st.caption(f"**GeekBrains — лидер в сфере онлайн-образования**")
            st.markdown(f"**Стек:** {' , '.join(data.loc[0]['stack'])}")

            # Вывод описания с использованием Markdown
            st.markdown(f"**Описание:**\n{data.loc[0]['summary']}")

            # Вывод дополнительной информации с использованием Markdown
            st.markdown(f"**Больше информации:** [{data.loc[0]['url']}]({data.loc[0]['url']})")

            st.write("---")

            st.subheader("Может быть интересно!")

            # Создаем две колонки для вывода описания курсов и картинок
            col1, col2, col3 = st.columns(3)

            # Функция для вывода информации о курсе и картинки
            def display_course_info(column, course_data):
                with column:
                    example = 'https://kursplus.ru/wp-content/uploads/stm_lms_avatars/stm_lms_avatar10.jpg?v=1645100697'
                    st.markdown(f'<a href="{course_data["url"]}"><img src="{example}" alt="Foo" width="150" height="150"/></a>', unsafe_allow_html=True)
                    st.markdown(f"**{course_data['name']}**")
                    score = course_data['total_score']
                    percent = 100 if score <= 0 else round(100 * (score / (score + 1)), 2)
                    st.markdown(f"<span style='color:green'>Схожесть: {percent}%</span>", unsafe_allow_html=True)
                    st.write(course_data['url'])

            # Вывод информации о втором курсе и картинке
            display_course_info(col1, data.loc[1])

            # Вывод информации о третьем курсе и картинке
            display_course_info(col2, data.loc[2])

            display_course_info(col3, data.loc[3])
            st.write("---")

            
            st.subheader("Верификация требований у дисциплин")
            txt = st.text_area(
            "⚙️ Киллер-фича!",
            "На основе резюме происходит поиск требований по вакансии "
            "эти требования сопоставляются с помощью NLP с результатами обучения в курсах "
            "результат можно посмотреть здесь или скачать "
            )

            matrix_df = get_matrix_df(vacancy_url, 'data/data_gb_resultv1.csv')

            st.dataframe(matrix_df, use_container_width=True)

            


    with tab2:
        vacancy_url = st.text_input("Введите ссылку на вакансию: ")
        resume_url = st.text_input("Введите ссылку на ваше резюме: ")

        # Кнопка для поиска схожего курса GeekBrains
        if st.button("Найти схожий курс от GeekBrains "):
            data, vacancy = preprocess_and_match_vacancy(vacancy_url=vacancy_url, data_gb_path='data/data_gb_resultv1.csv', data_it_path='data/preprocess_it.csv')

            finder = ResumeSkillsFinder()
            all_skills = []

            # Объединяем все массивы в один большой список
            for stack in data['stack']:
                all_skills.extend(stack)
            finder.skills = list(set(all_skills))

            def download_pdf(url, destination):
                response = requests.get(url)
                if response.status_code == 200:
                    with open(destination, 'wb') as f:
                        f.write(response.content)
                    return True
                else:
                    return False

            if download_pdf(resume_url, 'resume.pdf'):
                print("PDF downloaded successfully.")
            else:
                print("Failed to download PDF.")

            # Выводим описание вакансии
            st.markdown(f"**Название вакансии:** {vacancy['name']}")


            # Проверка наличия ключевых навыков и их вывод с использованием Markdown
            if vacancy['key_skills']:
                st.markdown(f"**Стек требований:** {' , '.join(vacancy['key_skills'])}")

            found_skills = finder.find_skills_in_resume('resume.pdf')
            st.markdown(f"**Стек пользователя:** {' , '.join(found_skills)}")

            vacancy['key_skills'] = list(set(vacancy['key_skills']) - set(found_skills))


            st.write("---")

            st.subheader("Наиболее подходящие курсы для прохождения:")
            st.markdown(f"**{data.loc[0]['name'].upper()}**")
            score = data.loc[0]['total_score']
            percent = 100 if score <= 0 else round(100 * (score / (score + 1)), 2)
            st.markdown(
                f"<span style='color:green'>Схожесть по переплексии описания и навыков: <b>{percent}%</b></span>",
                unsafe_allow_html=True
            )
            skills1 = [k.lower() for k in found_skills]
            skills2 =  list(set([k.lower() for k in found_skills] + [k.lower() for k in data.loc[0]['stack']]))
            difference = compare_models(skills1, skills2)
            st.markdown(
                f"<span style='color:green'>Прирост заработной платы после прохождения курса (рынок труда): <b>+{int(difference)} руб.</b></span>",
                unsafe_allow_html=True
            )
            st.markdown(f'<a href="{"https://raw.githubusercontent.com/ikanam-ai/VacancySkillsEstimator/main/logo.jpg"}"><img src="{"https://raw.githubusercontent.com/ikanam-ai/VacancySkillsEstimator/main/logo.jpg"}" alt="Foo" width="500" height="250"/></a>', unsafe_allow_html=True)
            st.caption(f"**GeekBrains — лидер в сфере онлайн-образовани**")
            st.markdown(f"**Стек:** {' , '.join(data.loc[0]['stack'])}")

            # Вывод описания с использованием Markdown
            st.markdown(f"**Описание:**\n{data.loc[0]['summary']}")

            # Вывод дополнительной информации с использованием Markdown
            st.markdown(f"**Больше информации:** [{data.loc[0]['url']}]({data.loc[0]['url']})")

            st.write("---")

            st.subheader("Может быть интересно!")

            # Создаем две колонки для вывода описания курсов и картинок
            col1, col2, col3 = st.columns(3)

            # Функция для вывода информации о курсе и картинки
            def display_course_info(column, course_data):
                with column:
                    example = 'https://kursplus.ru/wp-content/uploads/stm_lms_avatars/stm_lms_avatar10.jpg?v=1645100697'
                    st.markdown(f'<a href="{course_data["url"]}"><img src="{example}" alt="Foo" width="150" height="150"/></a>', unsafe_allow_html=True)
                    st.markdown(f"**{course_data['name']}**")
                    score = course_data['total_score']
                    percent = 100 if score <= 0 else round(100 * (score / (score + 1)), 2)
                    st.markdown(f"<span style='color:green'>Схожесть: {percent}%</span>", unsafe_allow_html=True)
                    skills1 = [k.lower() for k in found_skills]
                    skills2 =  list(set([k.lower() for k in found_skills] + [k.lower() for k in course_data['stack']]))
                    difference = compare_models(skills1, skills2)
                    st.markdown(
                        f"<span style='color:green'>Прирост заработной платы: <b>+{int(difference)} руб.</b></span>",
                        unsafe_allow_html=True
                    )
                    st.write(course_data['url'])



            # Вывод информации о втором курсе и картинке
            display_course_info(col1, data.loc[1])

            # Вывод информации о третьем курсе и картинке
            display_course_info(col2, data.loc[2])

            display_course_info(col3, data.loc[3])
            st.write("---")

            st.subheader("Верификация требований у дисциплин")
            txt = st.text_area(
            "⚙️ Киллер-фича!",
            "На основе резюме происходит поиск требований по ваканасии "
            "эти требования сопоставляются с помощью NLP с результатами обучения в курсах "
            "результат можно посмотреть здесь или скачать "
            )

            matrix_df = get_matrix_df(vacancy_url, 'data/data_gb_resultv1.csv')

            st.dataframe(matrix_df, use_container_width=True)

    with tab3:
        st.subheader('Кластерный анализ (t-SNE Visualization with Clusters) для датасета из файла it.csv')

if __name__ == "__main__":
    main()
