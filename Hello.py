import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка базы знаний кураторов
kb_data = pd.read_csv('curator_knowledge_base.csv', sep=';', encoding='utf-8')

print(kb_data)

# Функция для поиска наиболее подходящего ответа
def find_best_answer(question):
    # Векторизация вопроса и базы знаний
    vectorizer = TfidfVectorizer()
    question_vec = vectorizer.fit_transform([question])
    kb_vec = vectorizer.transform(kb_data['Question'])
    
    # Вычисление косинусной близости между вопросом и базой знаний
    similarity_scores = cosine_similarity(question_vec, kb_vec)
    print (similarity_scores)
    # Получение индекса наиболее похожего вопроса
    best_match_index = similarity_scores.argmax()
    
    # Проверка порога сходства
    similarity_threshold = 0.5  # Задайте подходящее значение порога
    if similarity_scores[0, best_match_index] < similarity_threshold:
        category = 'Other'
        answer = 'Redirect to Human'
    else:
        # Получение категории и ответа для наиболее похожего вопроса
        category = kb_data.iloc[best_match_index]['Category']
        answer = kb_data.iloc[best_match_index]['Answer']
    
    return category, answer

# Функция для обработки вопросов пользователя
def process_question(question):
    category, answer = find_best_answer(question)
    
    # Если категория не определена однозначно, задать уточняющий вопрос
    if category == 'Clarification Needed':
        clarification = st.text_input("Пожалуйста, уточните ваш вопрос:")
        if clarification:
            category, answer = find_best_answer(clarification)
    
    # Если ответ не найден, перенаправить запрос живому куратору
    if answer == 'Redirect to Human':
        st.write("Извините, я не могу ответить на ваш вопрос. Пожалуйста, обратитесь к живому куратору Тимофею.")
    else:
        st.write(answer)

# Интерфейс чат-бота
st.title("Чат-бот поддержки куратора1")

question = st.text_input("Введите ваш вопрос:")

if st.button("Отправить"):
    process_question(question)