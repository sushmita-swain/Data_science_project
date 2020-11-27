#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python -m pip install tweepy ')


# In[2]:


get_ipython().system('python -m pip install textblob')


# In[3]:


get_ipython().system('python -m textblob.download_corpora')


# In[4]:


import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
  
class TwitterClient(object): 
    ''' 
    Generic Twitter Class for sentiment analysis. 
    '''
    def __init__(self): 
        ''' 
        Class constructor or initialization method. 
        '''
        # keys and tokens from the Twitter Dev Console 
        consumer_key = 'bYfhXuthzv0EOJkmXZmGfZ3dl'
        consumer_secret = 'Dw5a9vJWpFsZzWUVTLKrpWk09BPIh3vUYHZxoif0C93IW6EJaG'
        access_token = '1221614283835858945-oIncNxhQmhcnm2avooFKhLSioHLyHK'
        access_token_secret = 'VpYHWynGjytnZgwUlbaF0hjw3QlyjwxaThyeIT50ssetN'
  
        # attempt authentication 
        try: 
            # create OAuthHandler object 
            self.auth = OAuthHandler(consumer_key, consumer_secret) 
            # set access token and secret 
            self.auth.set_access_token(access_token, access_token_secret) 
            # create tweepy API object to fetch tweets 
            self.api = tweepy.API(self.auth) 
        except: 
            print("Error: Authentication Failed") 
  
    def clean_tweet(self, tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+) | ('^https?:\/\/.*[\r\n]*')", " ", tweet).split()) 
  
    def get_tweet_sentiment(self, tweet): 
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(self.clean_tweet(tweet)) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'
  
    def get_tweets(self, query, count=3000): 
        ''' 
        Main function to fetch tweets and parse them. 
        '''
        # empty list to store parsed tweets 
        tweets = [] 
  
        try: 
            # call twitter api to fetch tweets 
            fetched_tweets = self.api.search(q=query, count=count) 
  
            # parsing tweets one by one 
            for tweet in fetched_tweets: 
                # empty dictionary to store required params of a tweet 
                parsed_tweet = {} 
  
                # saving text of tweet 
                parsed_tweet['text'] = tweet.text 
                # saving sentiment of tweet 
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 
  
                # appending parsed tweet to tweets list 
                if tweet.retweet_count > 0: 
                    # if tweet has retweets, ensure that it is appended only once 
                    if parsed_tweet not in tweets: 
                        tweets.append(parsed_tweet) 
                else: 
                    tweets.append(parsed_tweet) 
  
            # return parsed tweets 
            return tweets 
  
        except tweepy.TweepError as e: 
            # print error (if any) 
            print("Error : " + str(e)) 


# In[5]:



# creating object of TwitterClient Class 
api = TwitterClient() 
# calling function to get tweets 
tweets = api.get_tweets(query = 'Amazon India') 

# picking positive tweets from tweets 
ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
# percentage of positive tweets 
print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
# picking negative tweets from tweets 
ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
# percentage of negative tweets 
print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))

neuttweets = [tweet for tweet in tweets if tweet['sentiment'] == 'neutral'] 
# percentage of neutral tweets 
print("Neutral tweets percentage: {} %".format(100*(len(tweets) - len(ntweets) - len(ptweets))/len(tweets)))

print("\n\ntweets:")
for tweet in tweets[:5]: 
  print(tweet)

# printing first 5 positive tweets 
print("\n\nPositive tweets:") 
for tweet in ptweets[:10]: 
  print(tweet['text']) 

# printing first 5 negative tweets 
print("\n\nNegative tweets:") 
for tweet in ntweets[:10]: 
  print(tweet['text']) 

print("\n\nNeutral tweets:") 
for tweet in neuttweets[:10]: 
  print(tweet['text']) 
  
print(len(tweets), len(ntweets), len(ptweets), len(neuttweets))


# In[6]:


import pandas as pd

txt = []
sentiment = []
for tweet in tweets:
    txt.append(tweet['text'])
    sentiment.append(tweet['sentiment'])

df = pd.DataFrame(list(zip(txt, sentiment)), columns=['tweet_txt', 'tweet_sentiment'])
df


# In[7]:


flipkart_tweets = api.get_tweets(query = 'Flipkart')

fk_txt = []
fk_sentiment = []
for tweet in flipkart_tweets:
    fk_txt.append(tweet['text'])
    fk_sentiment.append(tweet['sentiment'])

df1 = pd.DataFrame(list(zip(fk_txt, fk_sentiment)), columns=['tweet_txt', 'tweet_sentiment'])
df1


# In[8]:


snapdeal_tweets = api.get_tweets(query = 'Snapdeal')

sd_txt = []
sd_sentiment = []
for tweet in snapdeal_tweets:
    fk_txt.append(tweet['text'])
    fk_sentiment.append(tweet['sentiment'])

df2 = pd.DataFrame(list(zip(sd_txt, sd_sentiment)), columns=['tweet_txt', 'tweet_sentiment'])
df2


# In[9]:


data = pd.concat([df, df1, df2]).reset_index(drop=True)
data


# In[10]:


from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# lower the tweets
data['preprocessed_tweet_txt'] = data['tweet_txt'].str.lower()

# filter out stop words and URLs
en_stop_words = set(stopwords.words('english'))
extended_stop_words = en_stop_words |                     {
                        '&amp;', 'rt',                           
                        'th','co', 're', 've', 'kim', 'daca'
                    }
url_re = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'        
data['preprocessed_tweet_txt'] = data['preprocessed_tweet_txt'].apply(lambda row: ' '.join([word for word in row.split() if (not word in extended_stop_words) and (not re.match(url_re, word))]))

# tokenize the tweets
tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
data['tokenized_tweet_txt'] = data['preprocessed_tweet_txt'].apply(lambda row: tokenizer.tokenize(row))
    

data


# In[11]:


# Bag of word

from sklearn.feature_extraction.text import CountVectorizer


# get most frequent words and their counts
def get_most_freq_words(str, n=None):
    vect = CountVectorizer().fit(str)
    bag_of_words = vect.transform(str)
    sum_words = bag_of_words.sum(axis=0) 
    freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
    freq =sorted(freq, key = lambda x: x[1], reverse=True)
    return freq[:n]
  
get_most_freq_words([ word for tweet in data.tokenized_tweet_txt for word in tweet],10)


# In[12]:


# finding the number of topics

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# build a dictionary where for each tweet, each word has its own id.
# We have 6882 tweets and 10893 words in the dictionary.
tweets_dictionary = Dictionary(data.tokenized_tweet_txt)

# build the corpus i.e. vectors with the number of occurence of each word per tweet
tweets_corpus = [tweets_dictionary.doc2bow(tweet) for tweet in data.tokenized_tweet_txt]

# compute coherence
tweets_coherence = []
for nb_topics in range(1,36):
    lda = LdaModel(tweets_corpus, num_topics = nb_topics, id2word = tweets_dictionary, passes=10)
    cohm = CoherenceModel(model=lda, corpus=tweets_corpus, dictionary=tweets_dictionary, coherence='u_mass')
    coh = cohm.get_coherence()
    tweets_coherence.append(coh)

# visualize coherence
plt.figure(figsize=(10,5))
plt.plot(range(1,36),tweets_coherence)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.show()


# In[13]:


# Running LDA

import matplotlib.gridspec as gridspec
import math


k = 6
tweets_lda = LdaModel(tweets_corpus, num_topics = k, id2word = tweets_dictionary, passes=10)

def plot_top_words(lda=tweets_lda, nb_topics=k, nb_words=10):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = gridspec.GridSpec(round(math.sqrt(k))+1,round(math.sqrt(k))+1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(20,15))
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='blue', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Topic "+str(i))
        
  
plot_top_words()


# In[14]:


# word cloud 

wd_corpus = []
for i in data.tokenized_tweet_txt:
    tmp = ' '.join(x for x in set(i))
    wd_corpus.append(tmp)

print(wd_corpus)


# In[15]:


#Word cloud
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
wordcloud = WordCloud(
                          max_words=30000,
                          max_font_size=50, 
                          random_state=42,
                          width=1600, height=800
                         ).generate(str(wd_corpus))
print(wordcloud)
fig = plt.figure(figsize=(30, 15), facecolor='k')
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)

plt.show()


# In[16]:


# train random-forest with tfidf vectorizer for sentement analysis

label = {
        'positive': 1,
        'negative': -1,
        'neutral': 0
}
data['tweet_sentiment'] = data['tweet_sentiment'].apply(lambda a: label[a])
data


# In[17]:


import seaborn as sns

# Distribution of the target variable
plt.figure(figsize=(30,10))
plt.xticks(fontsize=24, rotation=0)
plt.yticks(fontsize=24, rotation=0)
sns.countplot(data=data, x='tweet_sentiment')


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
tfv.fit(list(data['preprocessed_tweet_txt']))
# pickle.dump(tfv, open('data/tfv.pickle', 'wb'))


# In[19]:


x = tfv.transform(data['preprocessed_tweet_txt'])
y = data['tweet_sentiment'].values


# In[23]:


#tsne

from sklearn.manifold import TSNE
import numpy as np

def tsne_plot(x1, y1):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8,
                label='neutral')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8,
                label='positive')
    plt.scatter(X_t[np.where(y1 == -1), 0], X_t[np.where(y1 == -1), 1], marker='o', color='b', linewidth='1', alpha=0.8,
                label='negative')

    plt.legend(loc='best')
    plt.show()


# In[25]:


#tsne_plot(x, y)


# In[22]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.25)

clf = RandomForestClassifier().fit(train_x, train_y)
# pickle.dump(clf, open('data/rf_tfidf.pickle', 'wb'))
pred_y = clf.predict(val_x)

print(classification_report(val_y, pred_y))
print(accuracy_score(val_y, pred_y))


# In[ ]:




