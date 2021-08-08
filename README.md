# ai-covid-retweets

## Setup virtual environment
`virtualenv -p Python3 .`
`source bin/activate`
`pip install -r requirements.txt`

## Run flask api service
`FLASK_ENV=development FLASK_APP=app.py flask run`

## Save current dependencies
`pip3 freeze > requirements.txt`