import streamlit as st
import pandas as pd
import pythainlp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from collections import Counter


def featurize(text):
    words = pythainlp.word_tokenize(text)
    thai_dict = pythainlp.corpus.thai_words()
    word_counts = Counter(word for word in words if word in thai_dict)
    return dict(word_counts)

with open('sentiment_toxic_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

tokenizer = pythainlp.word_tokenize
stopwords = pythainlp.corpus.thai_stopwords()

# Set the app title
st.set_page_config(page_title='Thai Sentiment Analysis')


# Create the main app area
def rude_app():
    st.title('Thai Sentiment Analysis')
    text_input = st.text_area('Input text here:', height=200)
    if text_input:
    #     # Preprocess the input text
    #     preprocessed_text = ' '.join([word for word in tokenizer(text_input) if word not in stopwords])

    #     # Convert the preprocessed text into a bag-of-words representation
    #     text_bow = vectorizer.transform([preprocessed_text])

    #     # Use the trained model to predict sentiment score
    #     predicted_score = model.predict(text_bow)[0]

    #     weights = pd.DataFrame(model.coef_,  index=model.classes_)

    #     # Display the predicted sentiment score on the app
    #     st.write('Predicted sentiment score:', predicted_score, weights )

    ### new model ###

        feature_vector = featurize(text_input)
        print(feature_vector)
        # Convert the feature vector into a feature matrix
        X = vectorizer.transform(feature_vector)

        # Get the decision function values for the input text
        scores = model.decision_function(X)
        sentiment = model.predict(vectorizer.transform(feature_vector))[0]

        # Print the score for each word in the input text
        print(sentiment)
        for word, score in zip(feature_vector.keys(), scores):
            print(f'{word}: {score}')

        if sentiment == 'positive':
            color = 'green'
        elif sentiment == 'neutral':
            color = 'black'
        else:
            color = 'red'

        st.write('Sentiment: <b style="color:{}">{}</b>'.format(color, sentiment), unsafe_allow_html=True)
        st.write('Feature vector score: {}'.format(feature_vector), unsafe_allow_html=True)


rude_app()

