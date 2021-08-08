import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split

df = pd.read_csv('data/4class_categorised.csv')
df.drop('Unnamed: 0', axis='columns', inplace=True)

X = df[['followers','friends','favorites','mentions','hashtags','urls','sentistrength']]
X = sm.add_constant(X).values
Y = df['retweet_group'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# save numpy array as npy file
from numpy import save
# save to npy file
save('data/X_train.npy', X_train)
save('data/X_test.npy', X_test)
save('data/y_train.npy', y_train)
save('data/y_test.npy', y_test)