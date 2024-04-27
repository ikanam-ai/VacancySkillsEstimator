import streamlit as st

def main():
    st.title("Простое приложение на Streamlit")

    # Создаем поле для ввода текста
    user_input = st.text_input("Введите текст:")

    # Кнопка для отправки текста
    if st.button("Преобразовать в верхний регистр"):
        # Преобразуем текст в верхний регистр и выводим его
        result = user_input.upper()
        st.write("Текст в верхнем регистре:", result)

if __name__ == "__main__":
    main()
