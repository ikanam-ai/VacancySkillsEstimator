<p align="center">
    <img src="./logo.jpg" alt="Логотип проекта" width="500" style="display: inline-block; vertical-align: middle; margin-right: 10px;"/>  <br/>
     <H2 align="center">Команда Ikanam</H2> 
    <H2 align="center">Кейс "Компетентностный подбор образовательных курсов"</H2> 
</p>


> Команда Ikanam представляет жюри инновационный программный модуль с использованием ИИ для нахождения наиболее релевантных образовательных курсов GeekBrains на основе представленной вакансии с сайта HH.ru. Данный программный продукт является мощным инструментом для автоматизации анализа и улучшения качества пользовательского опыта.


## Установка и запуск

**Наше решение разделено на две ключевые части. Первая часть включает в себя подробное руководство по развёртыванию без необходимости взаимодействия с интерфейсом, что обеспечивает быстрое и эффективное тестирование. Вторая часть предоставляет подробные инструкции по развертыванию решения с использованием интерфейса, обеспечивая при этом максимальную удобство и интуитивную навигацию.**

***Часть 1:***
----------

*1. Загрузите репозиторий на свой компьютер и откройте её в вашей предпочитаемой среде разработки (IDE).* 

*2. Откройте терминал в IDE и введите туда следующую команду:* 

```python
python -m venv .venv
```
*3. Дождитесь создание папки `.venv` затем введите следующую команду:*

```python
.\.venv\Scripts\activate
```
*4. После активации установите все библиотеки (весрия python==3.10+) при помощи данной команды:*

```python
pip install -r requirements.txt
```

*5. Дождитесь установки всех библиотек и введите следующую команду.*

```python
pip install parse_hh_data==0.1.14
```

*6. Прекрасно! Теперь в терминале введите команду для перехода к основной папке с кодом проекта.*

```shell
cd src 
```

*7. Для запуска сервиса, введите следующую команду.*

```python
streamlit run src/app.py
```


<p align="center">
    <img src="./entry_streamlit_service.jpg" alt="Логотип проекта" width="900" style="display: inline-block; vertical-align: middle; margin-right: 10px;"/>  <br/>
    Начальная страница сервиса
</p>


# Пример использования сервиса, иллюстрированный процессом обработки скринкаста.


*Заполнить*


# Пример работы сервиса

*Заполнить*


Структура проекта
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │       └── build_features.py
    
--------

