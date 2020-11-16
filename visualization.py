import streamlit as st
import pickle
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt

stemmer = SnowballStemmer('english')
import re

vectorizer, pca, model = pickle.load(open('model.p', "rb"))
topic_word = pd.DataFrame(pca.components_.round(3),
             columns = vectorizer.get_feature_names())

def stemmit(word):
    return  stemmer.stem(re.sub(r'[^\w\s]', '', word))

def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            st.write("\nTopic ", str(ix))
        else:
            st.write("\nTopic: '",str(topic_names[ix]),"'")
        st.write(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


new_verse = st.text_input("Enter a verse")


if new_verse != "":
    words = new_verse.split(" ")
    words_stemmed = [stemmit(word) for word in words]
    in_stopwords = [(word in vectorizer.stop_words) for word in words_stemmed]
    not_stopword = [word for word in words if (stemmit(word) not in vectorizer.stop_words)&(stemmit(word) in vectorizer.vocabulary_)]

    vectorized = vectorizer.transform([" ".join(words_stemmed)])
    pca_d = pca.transform(vectorized)
    prediction = (model.predict(pca_d)[0])

    predict_map = {"p": "Priestly source", "d": "Deuteronomist", "y": "Yahwist"}
    st.subheader("Prediction: "+predict_map[prediction])

    slider = ""

    placeholder = st.empty()
    if len(not_stopword)>1:
        slider = st.select_slider("Select a word to view", options=not_stopword)
    else:
        slider = not_stopword[0]

    new_string = ""
    for i, w in enumerate(words):
        if stemmit(slider)==stemmit(w):
            new_string+="**"
        if not in_stopwords[i] and stemmit(w) in vectorizer.vocabulary_:
            new_string+='<span style="color:blue">'+w+'</span>'
        else:
            new_string+=w
        if stemmit(slider)==stemmit(w):
            new_string+="**"
        new_string+=" "
    placeholder.markdown(new_string, unsafe_allow_html=True)

    labels_ = ["Topic "+str(n) for n in range(10)]
    labels_ = ["Moses\n Speaks", "Command\n-ments", "Cleanliness", "Israel", "Sacrifices", "Prince of \n Egypt", "Lineages", "Time", "Command\n-ments ", "Egypt"]

    if stemmit(slider) in vectorizer.vocabulary_:
        slider_pca = pca.transform(vectorizer.transform([stemmit(slider)]))[0]
        #st.write(stemmit(slider))
        topic_model = topic_word[stemmit(slider)]
        fig =plt.figure(figsize=(8, 5))
        ax = fig.add_axes([0,0,1,1])
        plot = ax.bar(labels_[:len(topic_model)], height=topic_model)
        st.write(fig)
    display_topics(pca, vectorizer.get_feature_names(), 10, labels_)
