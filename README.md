# ai-covid-retweets

## Setup virtual environment
`virtualenv -p Python3 .`
`source bin/activate`
`pip install -r requirements.txt`

### Order of running
preprocess.py
random_forest_training.py


## Run flask api service
`FLASK_ENV=development FLASK_APP=app.py flask run`

## Save current dependencies
`pip3 freeze > requirements.txt`

## Saving models
1. Saving scalers
```
import joblib
scaler_filename = "scaler_x_1.save"
joblib.dump(scaler_x_1, scaler_filename) 

# And now to load...

scaler = joblib.load(scaler_filename) 
```
2. Saving keras model
```
# serialize model to JSON
model_json = model1.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model1.save_weights("model.h5")
print("Saved model to disk")

from keras.models import model_from_json

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
```