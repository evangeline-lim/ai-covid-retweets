from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
app = Flask(__name__)
api = Api(app)

import numpy as np
import pickle
import joblib
from tensorflow.python.keras.models import model_from_json

# load random forest model
with open('data/random_forest.pkl', 'rb') as f:
    clf = pickle.load(f)

# ------------------ MODEL 1 ------------------
# load neural network model
json_file = open('data/model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = model_from_json(loaded_model_json)
# load weights into new model
model1.load_weights('data/model1.h5')
# load scalers
scaler_x_1 = joblib.load('data/scaler_x_1.save')
scaler_y_1 = joblib.load('data/scaler_y_1.save')
# ------------------ MODEL 2 ------------------
# load neural network model
json_file = open('data/model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model2 = model_from_json(loaded_model_json)
# load weights into new model
model2.load_weights('data/model2.h5')
# load scalers
scaler_x_2 = joblib.load('data/scaler_x_2.save')
scaler_y_2 = joblib.load('data/scaler_y_2.save')
# ------------------ MODEL 3 ------------------
# load neural network model
json_file = open('data/model3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model3 = model_from_json(loaded_model_json)
# load weights into new model
model3.load_weights('data/model3.h5')
# load scalers
scaler_x_3 = joblib.load('data/scaler_x_3.save')
scaler_y_3 = joblib.load('data/scaler_y_3.save')
print("Loaded models from disk")

def predict_grp(test):
    y_pred = clf.predict(test)
    return y_pred[0]

def predict_retweet_grp1(test):
    test_scaled = scaler_x_1.transform(test)
    y_pred_scaled = model1.predict(test_scaled)
    y_pred = scaler_y_1.inverse_transform(y_pred_scaled)
    prediction = int(y_pred[0][0])
    return 'Final prediction: {}'.format(str(prediction))

def predict_retweet_grp2(test):
    test_scaled = scaler_x_2.transform(test)
    y_pred_scaled = model2.predict(test_scaled)
    y_pred = scaler_y_2.inverse_transform(y_pred_scaled)
    prediction = int(y_pred[0][0])
    return 'Final prediction: {}'.format(str(prediction))

def predict_retweet_grp3(test):
    test_scaled = scaler_x_3.transform(test)
    y_pred_scaled = model3.predict(test_scaled)
    y_pred = scaler_y_3.inverse_transform(y_pred_scaled)
    prediction = int(y_pred[0][0])
    return 'Final prediction: {}'.format(str(prediction))

class HelloWorld(Resource):
    def get(self):
        return {'Hello':'World!'}

class Test(Resource):
    def get(self):
        followers = request.args.get('followers', type=float)
        friends = request.args.get('friends', type=float)
        favorites = request.args.get('favorites', type=float)
        entities = request.args.get('entities', type=float)
        mentions = request.args.get('mentions', type=float)
        hashtags = request.args.get('hashtags', type=float)
        urls = request.args.get('urls', type=float)
        sentistrength = request.args.get('sentistrength', type=float)

        # string = '''Requested prediction for followers = {},
        # friends = {}, entities = {}, favorites = {}, mentions = {}, 
        # hashtags = {},' urls = {}, sentistrength = {}
        # '''.format(followers, friends, entities, favorites, mentions, hashtags, urls, sentistrength)

        query = np.array([followers, friends, entities, favorites, mentions, hashtags, urls, sentistrength]).reshape(1,-1)

        grp_prediction = predict_grp(query)
        if grp_prediction == 0:
            return '0 retweets'
        if grp_prediction == 1:
            return predict_retweet_grp1(query)
        if grp_prediction == 2:
            return predict_retweet_grp2(query)
        if grp_prediction == 3:
            return predict_retweet_grp3(query)

        return grp_prediction

api.add_resource(HelloWorld, '/')
api.add_resource(Test, '/test', endpoint='test')

# if __name__ == '__main__':
#     # run app in debug mode on port 5000
#     app.run(debug=True, port=5000)