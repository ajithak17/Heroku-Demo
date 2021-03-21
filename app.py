from flask import Flask,render_template,request
import requests,json

from werkzeug.utils import secure_filename
import os

import tensorflow as tf
import numpy as np
import pathlib

import requests
from io import BytesIO

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    return render_template('upload.html')

    


@app.route('/classify', methods=['POST'])
def classify():
    from tensorflow.keras.models import load_model
    model=load_model('model.h5')
    from tensorflow.keras.preprocessing import image
    if request.method == 'POST':
        # Get the file from post request
        f=request.files['file']
        path = request.form.get('hidden')
   
        classes=['maize','rice','sugarcane','wheat']

        def scale(image):
            image = tf.cast(image, tf.float32)
            image /= 255.0
            return tf.image.resize(image,[200,200])
        def decode_img(image):
            img = tf.image.decode_jpeg(image, channels=3)
            img = scale(img)
            return np.expand_dims(img, axis=0)
        
        content = requests.get(path).content
        result =np.argmax(model.predict(decode_img(content)),axis=1)
        return render_template('result.html', data=classes[result[0]],p=path)

     
  
        
if __name__ == '__main__':
    app.run()
