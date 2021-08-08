import pandas as pd

# Preprocess
dataset = pd.read_csv('./data/TweetsCOV19_052020.tsv', dtype=object, sep='\t')
dataset.columns = ["tweet_id", "username", "timestamp", "followers", "friends", "retweets", "favorites", "entities", "sentiment", "mentions", "hashtags", "urls"]

# Convert to int64
dataset[['followers', 'friends','retweets','favorites']]=dataset[['followers', 'friends','retweets','favorites']].apply(pd.to_numeric)

# Get SentiStrength
def get_sentistrength(arr):
  return (int(arr[0]) + abs(int(arr[1])))/10
dataset['sentistrength'] = dataset.apply(lambda row : get_sentistrength(row['sentiment'].split(' ')), axis = 1)
dataset.head()

# Count mentions/hashtags
def count_mentions_hashtags(s):
  if s == "null;":
    return 0
  else:
    return len(s.split(' '))
dataset['mentions'] = dataset.apply(lambda row : count_mentions_hashtags(str(row['mentions'])), axis = 1)
dataset['hashtags'] = dataset.apply(lambda row : count_mentions_hashtags(str(row['hashtags'])), axis = 1)

# Count URLs
dataset['urls'] = dataset.apply(lambda row : str(row['urls']).count(':-:'), axis = 1)

# Note:
# Handle repeat usernames?
# dataset['username'].value_counts()
# dataset.loc[dataset['username'] == '2435a45b85628172c5a47122144a7c67']

# Classify retweets into 0 or else
# dataset['retweet_group'] = np.where(dataset['retweets']== 0, 0, 1)
# dataset.to_csv('binary_categorised.csv')

# Classify retweets into 0 or else
def classify_retweets(retweets):
  if retweets == 0:
    return 0
  if (retweets < 10):
    return 1
  if (retweets < 100):
    return 2
  else:
    return 3

dataset['retweet_group'] = dataset.apply(lambda row : classify_retweets(row['retweets']), axis = 1)

dataset.to_csv('./data/4class_categorised.csv')
print(dataset.dtypes)
dataset.head()