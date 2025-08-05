def vectorize_text(X_train, X_val=None, X_test=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer()
    X_val_text_vectorized, X_test_text_vectorized = None, None
    X_train_text_vectorized = tfidf.fit_transform(X_train['comment_text'])
    if X_val is not None:
        X_val_text_vectorized = tfidf.transform(X_val['comment_text'])
    if X_test is not None:
        X_test_text_vectorized = tfidf.transform(X_test['comment_text'])
    return X_train_text_vectorized, X_val_text_vectorized, X_test_text_vectorized, tfidf

