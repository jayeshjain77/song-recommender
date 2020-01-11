from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.metrics.pairwise import linear_kernel
import numpy as np


app = Flask(__name__)

@app.route("/")
@app.route('/results')

def data():
	# here we want to get the value of user (i.e. ?user=some-value)
    query = request.args.get('query')
    print(query)	
    recommendations = recommender(query) 
    return recommendations.tostring()


def recommender(query):

	#load vocab and idfs and data
	#df = pd.read_csv("song_data.csv")
	vocabulary = pickle.load(open("vocabulary.pkl", "rb"))
	idfs = pickle.load(open("idf.pkl", "rb"))	
	tfidf_matrix = pickle.load(open("matrix.pkl","rb"))
	#tfidf_matrix = np.load("tfidf_matrix.npy")
	
	#reform the vectorizer
	tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
	tf.vocabulary_ = vocabulary
	tf.idf_ = idfs

	#query vector	
	vector = tf.transform([query])
	
	print(vector.shape)
	print(tfidf_matrix.shape)
	#similarity	
	cos_sim = linear_kernel(tfidf_matrix,vector)
	res = cos_sim[:,0].argsort()[:-6:-1]
	
	#prediction list
	#pred = [df['songname'][i] for i in res] 	
	
	return res
		

if __name__ == "__main__":
    app.run(host= '0.0.0.0')
