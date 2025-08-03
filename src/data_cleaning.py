import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
#--------------------------
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


#------------------------
df = pd.read_csv('data/raw/raw_dataset.csv')

def tokenize(text):
    return word_tokenize(text.lower())

stop_words = set(stopwords.words('english'))
def remove_stop_word(token):
    return [word for word in token if word not in stop_words]

lemmatizer = WordNetLemmatizer()
def lemmatize(token):
    return [lemmatizer.lemmatize(word) for word in token]

def remove_punctuation(token):
    output = []
    for word in token:
        word_clean = []
        for char in word:
            if char not in string.punctuation:
                word_clean.append(char)
        if len(word_clean) >  0:        
            output.append(''.join([char for char in word_clean]))
    return output

#------------------

def preprocessing(df):
    df['text1'] = df['comment_text'].astype(str).copy()
    df['text1'] = df['text1'].apply(tokenize)
    print('Finish tokenize')
    df['text1'] = df['text1'].apply(remove_punctuation)
    print('Finish remove punctuation')
    df['text1'] = df['text1'].apply(remove_stop_word)
    print('Finish remove_stop_word')
    df['text1'] = df['text1'].apply(lemmatize)
    print('Finish lemmatize')
    df['text'] = df['text1'].apply(lambda x: ' '.join(word for word in x))
    return df

df = preprocessing(df)
cleaned_df = df[['text', 'toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']]
print(cleaned_df.isna().sum())
cleaned_df = cleaned_df.dropna()
print(cleaned_df.isna().sum())
cleaned_df.to_csv('data/processed/cleaned_dataset.csv')



