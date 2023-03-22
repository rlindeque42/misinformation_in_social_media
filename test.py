from trigger_phrase import *

data = pd.read_csv("fake_news.csv")
tweet2 = data['text'][i]
tweet = 'Your priorities are our priorities. Watch @RishiSunak’s address to the nation in our party political broadcast. Tell Rishi what matters to you 👇'
cleanTweet(tweet2)