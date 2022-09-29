from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity, sigmoid_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

anime_features = pd.read_csv('anime_features.csv', delimiter = ',')
tfidf_matrix = tfidf.fit_transform(anime_features['genres_list'])

genre_df = pd.DataFrame(tfidf_matrix.toarray())
genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
genre_df.drop(columns='genre|unknown')
genre_df.reset_index(drop = True, inplace=True)

sig = sigmoid_kernel(genre_df, genre_df)

indicies = pd.Series(anime_features.index, index = anime_features['name']).drop_duplicates()

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/recommend/<recommendations>')
def recommend(recommendations):
    return render_template("result.html", animes = recommendations)

@app.route('/submit', methods = ['POST', 'GET'])
def give_rec(sig = sig):
    if request.method == 'POST':
        name = str(request.form['Anime'])
        idx = indicies[name]
        sig_scores = list(enumerate(sig[idx]))
        sig_scores = sorted(sig_scores, key = lambda x: x[1], reverse = True)
        sig_scores = sig_scores[1:11]
        movie_indicies = [i[0] for i in sig_scores]
        animes = list(anime_features['name'].iloc[movie_indicies])

    return redirect(url_for('recommend', recommendations = animes))

if __name__ == '__main__':
    app.run(debug=True)
