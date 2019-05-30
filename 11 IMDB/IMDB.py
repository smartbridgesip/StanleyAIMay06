# -*- coding: utf-8 -*-
"""
Created on Wed May 22 23:33:05 2019

@author: VSKUMAR
"""

#from importlib import reload
#import sys
#from imp import reload
#import warnings
#warnings.filterwarnings('ignore')
#if sys.version[0] == '2':
#    reload(sys)
#    sys.setdefaultencoding("utf-8")
import pandas as pd

df1 = pd.read_csv('labeledTrainData.tsv', delimiter="\t")
df1 = df1.drop(['id'], axis=1)
df1.head()

df2 = pd.read_csv('imdb_master.csv',encoding="latin-1")
df2.head()

df2 = df2.drop(['Unnamed: 0','type','file'],axis=1)
df2.columns = ["review","sentiment"]
df2.head()

df2 = df2[df2.sentiment != 'unsup']
df2['sentiment'] = df2['sentiment'].map({'pos': 1, 'neg': 0})
df2.head()

df = pd.concat([df1, df2]).reset_index(drop=True)
df.head()


import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))
df.head()
df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()

#len(sequence)
#import matplotlib.pyplot as plt
#leng=0
#length = [(leng + len(x)) for x in df['Processed_Reviews']]
#plt.hist(length)
#plt.xlabel('length of words')
#plt.ylabel('frequency')
#plt.savefig('im1.jpg')
#plt.boxplot(length, autorange = True)
#plt.title('Frequency Disturbution')
#plt.savefig('im2.jpg')


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])

maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = df['sentiment']

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 3
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

df_test=pd.read_csv("testData.tsv",header=0, delimiter="\t", quoting=3)
df_test.head()
df_test["review"]=df_test.review.apply(lambda x: clean_text(x))
df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
y_test = df_test["sentiment"]
list_sentences_test = df_test["review"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
prediction = model.predict(X_te)
y_pred = (prediction > 0.5)
from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)

def tokengen(inp):
    input_token = tokenizer.texts_to_sequences(inp)
    input_token
    inp_pred = pad_sequences(input_token, maxlen=maxlen)
    inp_pred
    prediction = model.predict(inp_pred)
    pred = (prediction > 0.5)
    return pred

inp = ["The first two Terminator films are among the best of their genre and are brilliant films also in their own right. Rise of the Machines started the series' decline and had its faults but also had some good things that made it better than its reputation, while Terminator Salvation although with its moments was even more disappointing. While not a complete disaster, the latest in the Terminator franchise is the worst yet and sees a once classic series at rock bottom. Are there any good points? Not many, but yes there are. The best thing about Terminator Genisys is Arnold Schwarzenegger, while more subdued than he can be he is rock-solid and has the most intensity and charisma of the entire cast. Despite not having a lot of screen time and being mute, Byung-hun Lee is nonetheless credible as T-1000. Some of the scenery is quite atmospheric and striking. Everything else was severely wanting. The rest of the cast are not good at all, with Jai Courtney being by far the worst actor for Kyle Reese of the series in a performance that is both insipid and annoying (which didn't entirely surprise me seeing as out of the little I've seen of him he's always struck me as a lazy actor with an arrogant ego). Just as much as the anaemic Emilia Clarke faring the worst of all the actresses playing Sarah Connor, who behaves in a way that you'd expect a stereotypical bratty and vapid high school teenager to act but not Sarah Connor. Jason Clarke is not quite as stiff as Christian Bale was in Terminator Salvation, but it is still a shock to see the role of John Connor being performed and written so blandly. The chemistry between them is also barely there, most of the time it's even non-existent. JK Simmons is basically wasted, and his dialogue and character are forced and out of place. The characters are like personality-less ciphers, with none of what made them so memorable before as characters and no convincing conflict, in a film that definitely could have benefited from less characters and more development (which is non-existent) . The script is overstuffed and confused, as well as tonally unbalanced with overly-complex scientific jargon, cheesy one-liners (even Schwarzenegger's don't work particularly well this time), misplaced comedy (that's even more distracting than in Rise of the Machines and used with even less subtlety) and too many ideas barely explored. The story, as well as having a stitched-together episodic feel, is at best a head-scratcher, with it being confused to being at times incoherent as a result of doing too little with too much. It's also very dull, not just because of the leaden pacing but because there is not much new and little interesting is done with the ideas presented in the film and it completely lacks atmosphere, thrills, mystery or suspense, and any drama is both heavy handed and lacking in heart.Direction from Alan Taylor is lazy, favouring spectacle over depth and story and character development. And unfortunately the spectacle is not that good. The special effects are not terrible, but they at best never rise above just-passable (the worst of the series in this regard, they are quite poor actually in the beginning section and fake-looking too in other parts), are used with little to no imagination, and there is too many of them (sometimes in places where they were not even needed). The action sequences are equally painfully unimaginative, are sloppily edited, are leadenly and predictability choreographed and contain no tension or thrills whatsoever, let alone any fun. It's very erratically shot and scored with a mix of overbearing bombast and dirge-like drone. The ending also felt incredibly forced and tacked on."]

tokengen(inp)

