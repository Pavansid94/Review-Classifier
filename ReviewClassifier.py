import numpy as np
import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn import model_selection,naive_bayes,svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import accuracy_score,classification_report

np.random.seed(500) #important!
corpus = pd.read_csv('./amazon_review_corpus.csv',encoding='latin-1')

def get_wn_pos(word): #Using short tags for POS
    tag = nltk.pos_tag([word])[0][1][0].upper() #[('Fortunately', 'RB')]
    print(nltk.pos_tag([word]))
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag,wordnet.NOUN)

#print(get_wn_pos('Fortunately'))

#Pre-Processing
corpus['text'].dropna(inplace=True) #Remove blank entries
corpus['text'] = [entry.lower() for entry in corpus['text']] #to lowercase
corpus['text'] = [word_tokenize(entry) for entry in corpus['text']] #tokenize

for index,entry in enumerate(corpus['text']):
    #print(entry)
    final_text = []
    lemmatizer = WordNetLemmatizer()
    for word,tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha:
            lemmatized_word = lemmatizer.lemmatize(word,get_wn_pos(word))
            #print(lemmatized_word)
            #break
            final_text.append(lemmatized_word)
    corpus.loc[index,'text_final'] = str(final_text)

print(corpus.columns)
Train_X,Test_X,Train_Y,Test_Y = model_selection.train_test_split(corpus['text_final'],corpus['label'],test_size = 0.3) #70/30 split
print(corpus['label'].unique()) #unique labels

#convert labels to 0/1
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
#print(Train_Y," ",Test_Y)

#convert text into Tftdf vectors to maximum 2000 unique words
Tfidf_vect = TfidfVectorizer(max_features = 2000)
Tfidf_vect.fit(corpus['text_final'])
#print(Tfidf_vect.get_feature_names())
Train_X_Tfidf = Tfidf_vect.fit_transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#Using count vectorizer
Count_vect = CountVectorizer(analyzer='word',ngram_range = (1,3))
Count_vect.fit(corpus['text_final'])
print(Count_vect.get_feature_names())
Train_X_Count = Count_vect.fit_transform(Train_X)
Test_X_Count = Count_vect.transform(Test_X)

print(len(Tfidf_vect.vocabulary_))
#print(len(Count_vect.vocabulary_))

#checking with naive bayes
NB = naive_bayes.MultinomialNB()
NB.fit(Train_X_Tfidf,Train_Y)

predictions_NB = NB.predict(Test_X_Tfidf)
print("NB Score: ",accuracy_score(predictions_NB,Test_Y))

#using Tfidf with 'word' analyzer and max_df set to 0.5
Tfidf_vect_vocab = TfidfVectorizer(analyzer='word', max_df=0.5,stop_words='english')
Tfidf_vect_vocab.fit(corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.fit_transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#predict for the foll. text
vector = Tfidf_vect.transform([" Excellent Soundtrack: I truly like this soundtrack and I enjoy video game music. I have played this game and most of the music on here I enjoy and it's truly relaxing and peaceful.On disk one. my favorites are Scars Of Time, Between Life and Death, Forest Of Illusion, Fortress of Ancient Dragons, Lost Fragment, and Drowned Valley.Disk Two: The Draggons, Galdorb - Home, Chronomantique, Prisoners of Fate, Gale, and my girlfriend likes ZelbessDisk Three: The best of the three. Garden Of God, Chronopolis, Fates, Jellyfish sea, Burning Orphange, Dragon's Prayer, Tower Of Stars, Dragon God, and Radical Dreamers - Unstealable Jewel.Overall, this is a excellent soundtrack and should be brought by those that like video game music.Xander Cross"])
#vector = Tfidf_vect.transform([" Eight Crazy Nights: I couldn't tell you how this movie ends, we turned it off after the first half-hour. Not appropriate for children, not entertaing for adults. Our dog even left the room. Don't waste your money."])
NB = naive_bayes.MultinomialNB()
NB.fit(Train_X_Tfidf,Train_Y)
print(NB.predict(vector))

#SVM classifier, testing for accuracy
SVM = svm.SVC(C=1.0,kernel = 'linear', degree = 6, gamma = 0.7)
SVM.fit(Train_X_Tfidf,Train_Y)

predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Score: ",accuracy_score(predictions_SVM,Test_Y))

#Using SVM to predict a new txt
SVM = svm.SVC(C=1.0,kernel='linear')
SVM.fit(Train_X_Tfidf,Train_Y)
vector = Tfidf_vect.transform(list("Excellent Soundtrack: I truly like this soundtrack and I enjoy video game music. I have played this game and most of the music on here I enjoy and it's truly relaxing and peaceful.On disk one. my favorites are Scars Of Time, Between Life and Death, Forest Of Illusion, Fortress of Ancient Dragons, Lost Fragment, and Drowned Valley.Disk Two: The Draggons, Galdorb - Home, Chronomantique, Prisoners of Fate, Gale, and my girlfriend likes ZelbessDisk Three: The best of the three. Garden Of God, Chronopolis, Fates, Jellyfish sea, Burning Orphange, Dragon's Prayer, Tower Of Stars, Dragon God, and Radical Dreamers - Unstealable Jewel.Overall, this is a excellent soundtrack and should be brought by those that like video game music.Xander Cross"))
print(vector)
#print(SVM.predict(vector))