import time
import json
import pandas as pd
import csv
from twython import Twython

def hydrate(file_name, df, id_name):
    """
    This function hydrates the csv file of tweets that I input and saves it as a json file to the inputted file name
    """

    # Authenticating my twitter
    TWITTER_AUTH = {
        'app_key': '3OGjcflG8y5LqBXlAq2dkPJ2M',
        'app_secret': 'z5OmDSGlPcB8EYUrtcTb3S3BRHCOrd6uSX1NM5w9nwhPgCVKCC',
        'oauth_token': '1591178743757905932-uEt722NcYgrInQkLOmVSleyxnbaPdC',
        'oauth_token_secret': 'QXuOVYu10jtqxZUANyqhgG4ToMXQYgdGrvqwjcQacd9qo'
    }
        
    twitter = Twython(app_key=TWITTER_AUTH['app_key'],
                        app_secret=TWITTER_AUTH['app_secret'],
                        oauth_token=TWITTER_AUTH['oauth_token'],
                        oauth_token_secret=TWITTER_AUTH['oauth_token_secret'])

    # Cannot make too many requests so must divide into chunks
    def divide_chunks(ids, n):
        for i in range(0, len(ids), n):
            yield ids[i:i + n]

    tweet_ids = []

    for i in df.index:
        tweet_id = (df[id_name][i])
        tweet_ids.append((str(tweet_id).split("\t"))[0])

    tweet_ids = tweet_ids[1:]

    df_ids = pd.DataFrame(tweet_ids, columns=[id_name])

    chunks = list(divide_chunks(df_ids[id_name].values, 100))
    total_chunks = len(chunks)


    # Opening a file to save fake tweets to
    i = 0
    with open(file_name, 'a+', encoding='utf-8') as out_f: 
        for chunk in chunks:
            # Convert ID from integer to string
            chunk = [str(id) for id in chunk]
            i += 1
        
            # Check if gone over max requests
            if i % 175 == 0:
                print("i'm sleeping")
                time.sleep(60 * 15)

            while True:
                try:
                    # Get the tweets
                    search_results = twitter.lookup_status(id=','.join(chunk), map="false", trim_user="false", include_entities="true", tweet_mode="extended")
                except Exception as e:
                    # Check for error and attempt again
                    print(e)
                    sec = 60 * 10
                    print(f'Waiting {sec} second(s)...')
                    print()

                    time.sleep(sec)
                    continue

                break

            for tweet in search_results:    
                out_f.write(json.dumps(tweet))
                out_f.write('\n')

            print(f'Chunk {i} of {total_chunks} ({(i/total_chunks)*100}%)')

# Fetching datasets of the FNN tweets
df_fnn_fake = pd.read_csv('gossipcop_fake.csv', names=['id','news_url','title','tweet_ids'])
df_fnn_real = pd.read_csv('gossipcop_real.csv', names=['id','news_url','title','tweet_ids'])

# Fetching datasets of the ANTIVax tweets
df_anti = pd.read_csv('VaxMisinfoData.csv', names=['id','is_misinfo'])

# Running the hydrator 
hydrate("gossip_real_tweets", df_fnn_real)
hydrate("gossip_fake_tweets", df_fnn_fake)
hydrate("anti_tweets.json", df_anti)

# Making gossipcop tweets into 1 dataset and only include tweets and class label and converting to csv file
with open('gossip_real_tweets', 'r')as file:
    df = pd.read_json(file.read())

df_real = df['full_text']
df_real['is_real'] = [1]*len(df_real.index)

with open('gossip_fake_tweets', 'r')as file:
    df = pd.read_json(file.read())

df_fake = df['full_text']
df_fake['is_real'] = [0]*len(df_real.index)

df_gossip = df_fake.append(df_real)

df_gossip.to_csv('gossip_tweets.csv', index=True)

# Converting anti vax tweets to csv file
with open('anti_tweets.json', 'r')as file:
    df = pd.read_json(file.read())

df_vax = df['full_text']
df_id = df['id']

# This hydrator doesn't store the fake/real tag so need to run through the VaxMisinfoData and append tag
# Need to invert the tag from 'is_misinfo' to 'is_real'
df_vax_og = pd.read_csv('VaxMisinfoData.csv')



