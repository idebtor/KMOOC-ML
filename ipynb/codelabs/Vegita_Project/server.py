# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

import datetime
import tensorflow as tf
import numpy as np

app = Flask(__name__)

#저장된 모델 불러오기
model = tf.keras.models.load_model('./model/saved.ckpt')

@app.route("/", methods=['GET', 'POST'])
def index():
	if request.method == 'GET':
		return render_template('index.html')
	if request.method == 'POST':
		avg_temp = float(request.form['avg_temp'])
		min_temp = float(request.form['min_temp'])
		max_temp = float(request.form['max_temp'])
		rain_fall = float(request.form['rain_fall'])
		
	price = 0
	
	data = ((avg_temp, min_temp, max_temp, rain_fall), )  # 기존의 학습된 데이터와 같은 2차원 배열 만들기
	arr = np.array(data, dtype=np.float32)
	
	# 예측 수행
	x_data = arr[0:4]  # avg_temp, min_temp, max_temp, rain_fall
	
	price = model.predict(x_data)
	return render_template('index.html', price = price)

if __name__ == '__main__':
	app.run(debug=True)
