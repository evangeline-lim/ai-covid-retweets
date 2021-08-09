import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import train_test_split

df = pd.read_csv('data/4class_categorised.csv')
df.drop('Unnamed: 0', axis='columns', inplace=True)

# split df by groups
df_nort = df.loc[df['retweet_group']==0]
df_smallrt = df.loc[df['retweet_group']==1]
df_midrt = df.loc[df['retweet_group']==2]
df_manyrt = df.loc[df['retweet_group']==3]

df_nort.to_csv('data/no_retweets.csv')
df_smallrt.to_csv('data/1_9_retweets.csv')
df_midrt.to_csv('data/10_99_retweets.csv')
df_manyrt.to_csv('data/100_retweets.csv')

X = df[['followers','friends','favorites','mentions','hashtags','urls','sentistrength']]
X = sm.add_constant(X).values
Y = df['retweet_group'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# save to npy file
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)