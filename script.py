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

def overall_prediction(test):
    grp_prediction = predict_grp(query)
    if grp_prediction == 0:
        print('0 retweets')
        return '0 retweets'
    if grp_prediction == 1:
        print(predict_retweet_grp1(query))
        return predict_retweet_grp1(query)
    if grp_prediction == 2:
        print(predict_retweet_grp2(query))
        return predict_retweet_grp2(query)
    if grp_prediction == 3:
        print(predict_retweet_grp3(query))
        return predict_retweet_grp3(query)

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

print("Please specify the following inputs. Note: sentistrength is a float in range 0-1")

followers = input("Number of followers: ")
friends = input("Number of friends: ")
favorites = input("Number of favourites: ")
entities = input("Number of entities: ")
mentions = input("Number of mentions: ")
hashtags = input("Number of hastags: ")
urls = input("Number of urls: ")
sentistrength = input("Sentistrength: ")

if followers == '':
    followers = 0
if friends == '':
    friends = 0
if favorites == '':
    favorites = 0
if entities == '':
    entities = 0
if mentions == '':
    mentions = 0
if hashtags == '':
    hashtags = 0
if urls == '':
    urls = 0
if sentistrength == '':
    sentistrength = 0

query = np.array([float(followers), float(friends), float(entities), float(favorites), float(mentions), float(hashtags), float(urls), float(sentistrength)]).reshape(1,-1)
print('Input query', query)

overall_prediction(query)