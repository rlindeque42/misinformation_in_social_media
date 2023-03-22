import pandas as pd
import sys
import numpy as np
import string
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def remove_url(data):
    """
    Removing urls from the tweets
    """

    data['full_text'] = [re.sub(r"http\S+", "", str(x)) for x in data['full_text']]
    data['full_text'] = [re.sub(r"www.\S+", "", str(x)) for x in data['full_text']]

    return data


def remove_punct_noneng(data):
    """
    Removing punctuation, hashtags, mentions and non-English (aka non-ASCII) characters
    """

    punctuation_remove = string.punctuation
    punctuation_remove = punctuation_remove.replace('@', '')
    punctuation_remove = punctuation_remove.replace('#', '')
    data['full_text'] = data['full_text'].str.replace('[{}]'.format(punctuation_remove), '')
    list_to_remove = ["\r", "\n", "–","…", "•"]

    data['full_text'] = [re.sub(r"#\w+", "", str(x)) for x in data['full_text']]
    data['full_text'] = [re.sub(r"@\w+", "", str(x)) for x in data['full_text']]
    data['full_text'] = [re.sub("—", " ", str(x)) for x in data['full_text']] #replace - with space
    data["full_text"] = [re.sub('\s+', ' ', str(x)) for x in data["full_text"]]   #remove more than 2 consec spaces with just one space

    for elem in list_to_remove:
        data["full_text"] = data["full_text"].str.replace(elem, "")
    """
    for i in data.index:
        encoded_string = data['full_text'][i].encode("ascii", "ignore")
        data['full_text'][i] = encoded_string.decode()
    """
    return(data)

def remove_stop(data):
    """
    Removing stop words from the tweet
    """
    stop_words = set(stopwords.words('english'))

    for i in data.index:

        word_tokens = word_tokenize(data['full_text'][i])

        filtered_sentence = [w.lower() for w in word_tokens if w.lower() not in stop_words]

        data['full_text'][i] = TreebankWordDetokenizer().detokenize(filtered_sentence)

    return data

def remove_emoji(data):
    """
    Removing emojis from the tweet
    This code was taken from: https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                           "]+", flags=re.UNICODE)

    for i in data.index:
        data['full_text'][i] = emoji_pattern.sub(r'', data['full_text'][i])
    
    data['full_text'] = [re.sub(r"⁦", "", str(x)) for x in data['full_text']]
    data['full_text'] = [re.sub(r"⁩", "", str(x)) for x in data['full_text']]

    return data


# Cleaning gossip tweets
data = pd.read_csv("gossip_tweets.csv")

print("Doing data cleanup for GossipCop")

data = remove_url(data)
print("After removing URLs\n")

data = remove_punct_noneng(data)
print("After removing punctuations\n")

data = remove_stop(data)
print("After removing Stopwords\n")

data = remove_emoji(data)
print("After removing emojis")

data.to_csv("gossip_tweets_clean.csv", index = False)

# Cleaning Covid-19 tweets
data = pd.read_csv("vax_misinfo_tweets.csv")
print(data['is_real'][:1])

print("Doing data cleanup for ANTIVax")

data = remove_url(data)
print("After removing URLs\n")

data = remove_punct_noneng(data)
print("After removing punctuations\n")

data = remove_stop(data)
print("After removing Stopwords\n")

data = remove_emoji(data)
print("After removing emojis")

data.to_csv("vax_misinfo_tweets_clean.csv", index = False)

# Combining them into 1 dataset
df1 = (pd.read_csv('vax_misinfo_tweets_clean.csv'))[['full_text', 'is_real']]
df2 = (pd.read_csv('gossip_tweets_clean.csv'))[['full_text', 'is_real']]

df = df1.append(df2)
df.to_csv('fake_news_id.csv', index=False)

# Had to remove and re add the id in order for it to be sequential
df5 = pd.read_csv('fake_news_id.csv')
df5.to_csv('fake_news_label.csv', index_label='id')

# Had to flip the labels and rename the headers
df6 = pd.read_csv('fake_news_label.csv')
df6.rename({'full_text': 'text', 'is_real': 'label'}, axis=1, inplace=True)

for i in df6.index:
    df6['label'][i] = 1- df6['label'][i]

# Drop na values
df6 = df6.dropna()

df6.to_csv('fake_news.csv', index=False)

