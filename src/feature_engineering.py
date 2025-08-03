import pandas as pd
from sklearn.model_selection import train_test_split

#---------------------

df = pd.read_csv('data/processed/cleaned_dataset.csv')
df = df.dropna()
X, y = df[['text']], df.drop(columns=['text'])
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.2)
print(X_train.isna().sum())
#----------------------

def vectorize_text(X_train, X_val=None, X_test=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer()
    X_val_text_vectorized, X_test_text_vectorized = None, None
    X_train_text_vectorized = tfidf.fit_transform(X_train['text'])
    if X_val is not None:
        X_val_text_vectorized = tfidf.transform(X_val['text'])
    if X_test is not None:
        X_test_text_vectorized = tfidf.transform(X_test['text'])
    return X_train_text_vectorized, X_val_text_vectorized, X_test_text_vectorized

print(vectorize_text(X_train, X_test, X_val))