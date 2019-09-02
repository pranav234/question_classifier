from flask import *;
import numpy as np;
from keras.models import load_model;
from keras import backend as K
app = Flask(__name__)

import pickle
with open('emb_mat','rb') as pickle_file:
	emb_mat = pickle.load(pickle_file)


def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:30]
    embeds = [emb_mat.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (30 - len(embeds))
    return np.array(embeds)


@app.route("/")
def home():
	return render_template('home.html')

@app.route('/predict' , methods = ['POST'])
def predict():
	model = load_model('model.h5')

	if request.method == 'POST':
		question = request.form['question']
		array_text= np.array([text_to_array(question)])
		result = model.predict(array_text)
		K.clear_session()
		print(result)
	return render_template('result.html',result = result)	



