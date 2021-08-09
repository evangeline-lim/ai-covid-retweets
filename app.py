from flask import Flask, jsonify, request
from flask_restful import Resource, Api
app = Flask(__name__)
api = Api(app)

import numpy as np
import pickle
import joblib
from tensorflow.python.keras.models import model_from_json

# load random forest model
with open('data/random_forest.pkl', 'rb') as f:
    clf = pickle.load(f)

# load neural network model
json_file = open('data/model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = model_from_json(loaded_model_json)
# load weights into new model
model1.load_weights('data/model1.h5')
print("Loaded model from disk")

# load scalers
scaler_x_1 = joblib.load('data/scaler_x_1.save')
scaler_y_1 = joblib.load('data/scaler_y_1.save')

@app.route('/')
def hello_world():
    return jsonify({'Hello':'World!'})
    # return 'Hello, World!'

@app.route('/query-example')
def query_example():
    followers = request.args.get('followers')
    print(type(followers))
    return followers
    # return '''<h1>The followers value is: {}</h1>'''.format(followers)
    # return 'Query String Example'

'''X: followers, friends, favorites, mentions, hashtags, urls, sentistrength'''
query = np.array([0,0,0,0,0,0,0,0])
@app.route('/hi')
def predict_group():
    test = query.reshape(1,-1)
    y_pred = clf.predict(test)
    return 'Retweet class: ' + str(y_pred[0])

    # # evaluate loaded model on test data
    # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = loaded_model.evaluate(X, Y, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    # return str(score[1]*100)


def predict_retweet():
    test = query.reshape(1,-1)
    print(test)
    test_scaled = scaler_x_1.transform(test)
    print(test_scaled)
    y_pred_scaled = model1.predict(test_scaled)
    print(y_pred_scaled)
    y_pred = scaler_y_1.inverse_transform(y_pred_scaled)
    print(y_pred)
    prediction = int(y_pred[0][0])
    print(prediction)
    return 'Final prediction: {} \n'.format(str(prediction))

# if __name__ == '__main__':
#     # run app in debug mode on port 5000
#     app.run(debug=True, port=5000)