import time
import json
import pandas as pd
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
hydrate("vax_misinfo_tweets", df_anti)

# Combining them into 1 dataset
df1 = (pd.read_csv('gossipcop_fake.csv'))[['full_text', 'is_real']]
df2 = (pd.read_csv('gossip_tweets_clean.csv'))[['full_text', 'is_real']]

df = (df1.append(df2)).append(df3)
df6 = pd.read_csv('fake_news_dataset_short.csv')