import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return "hi"




if __name__ == "__main__":
    app.run(debug=True)
