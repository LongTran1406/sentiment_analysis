import mlflow
import mlflow.sklearn
import pandas as pd
from src import feature_engineering
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.naive_bayes import MultinomialNB
import pickle
#---------------

df = pd.read_csv('data/processed/cleaned_dataset.csv')
df = df.dropna()
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
best_f1 = -1
best_run_id = None

#--------------------
mlflow.set_tracking_uri("http://54.79.130.241:5000/")
for threshold in thresholds:
   with mlflow.start_run(run_name=f"threshold_{threshold}") as run:
        X = df[['text']]
        y = df['toxicity'].apply(lambda x: 1 if x >= threshold else 0)
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.2, stratify=y_train_val)
        X_train_text_vectorized, X_val_text_vectorized, X_test_text_vectorized, vectorizer = vectorize_text(X_train, X_val, X_test)
    
        model = MultinomialNB()
        model.fit(X_train_text_vectorized, y_train)
        
        val_f1 = f1_score(y_val, model.predict(X_val_text_vectorized))
        
        #--------------------------
        
        mlflow.log_metric("val_f1", f1_score(y_val, model.predict(X_val_text_vectorized)))
        mlflow.log_metric("val_recall", recall_score(y_val, model.predict(X_val_text_vectorized)))
        mlflow.log_metric("val_precision", precision_score(y_val, model.predict(X_val_text_vectorized)))
        mlflow.log_metric("val_roc_auc", roc_auc_score(y_val, model.predict_proba(X_val_text_vectorized)[:, 1]))
        mlflow.log_metric("val_avg_precision", average_precision_score(y_val, model.predict_proba(X_val_text_vectorized)[:, 1]))
        mlflow.log_param("threshold", threshold)
        
        #--------------------------
    
        mlflow.log_metric("test_f1", f1_score(y_test, model.predict(X_test_text_vectorized)))
        mlflow.log_metric("test_recall", recall_score(y_test, model.predict(X_test_text_vectorized)))
        mlflow.log_metric("test_precision", precision_score(y_test, model.predict(X_test_text_vectorized)))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, model.predict_proba(X_test_text_vectorized)[:, 1]))
        mlflow.log_metric("test_avg_precision", average_precision_score(y_test, model.predict_proba(X_test_text_vectorized)[:, 1]))

        #-------------------------
        
        mlflow.sklearn.log_model(model, "model")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_run_id = run.info.run_id
            best_model = model
            best_vectorizer = vectorizer
        
if best_run_id is not None:
    client = mlflow.tracking.MlflowClient()
    model_url = f"runs:/{best_run_id}/model"
    model_name = "best_mnb_model"
    
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass
    
    model_version = client.create_model_version(
        name = model_name,
        source = model_url,
        run_id = best_run_id
        )        
    print("Register success")   
    
    #--------------------
    
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("best_vectorizer.pkl", "wb") as f:
        pickle.dump(best_vectorizer, f)

        
        
        
        