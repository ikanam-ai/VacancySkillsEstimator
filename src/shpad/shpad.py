import pandas as pd
from catboost import CatBoost

model = CatBoost()
model.load_model('catboost_model_all.cbm')

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_skills(skills, column_names):
    skills = [i.lower() for i in skills]
    skills_vector = []
    for column in column_names:
        if column in skills:
            skills_vector.append(1)
        else:
            skills_vector.append(0)
    return skills_vector

def predict_result(model, variables_vector, skills_vector):
    res_vector = variables_vector + skills_vector
    return round(float(model.predict([res_vector])), 2)

def compare_models(skills1, skills2):
    variables_vector = ['fullDay', 'between1And3', 'full', 'Москва', 'Центральный']

    column_names = pd.read_csv('df_hh_res_all.csv').columns[6:]
    skills_vector1 = preprocess_skills(skills1, column_names)
    skills_vector2 = preprocess_skills(skills2, column_names)
    prediction1 = predict_result(model, variables_vector, skills_vector1)
    prediction2 = predict_result(model, variables_vector, skills_vector2)
    return round(abs(prediction1 - prediction2), 2)

df_hh_res = load_data('df_hh_res_all.csv')
column_names = df_hh_res.columns[6:]
skills1 = ['Python', 'SQL', 'PyTorch', 'NLP', 'Git', 'Docker', 'Linux', 'C++']
skills2 = ['Python', 'SQL', 'PyTorch', 'NLP', 'Git', 'Docker', 'Linux', 'C++', 'JavaScript']
difference = compare_models(skills1, skills2)