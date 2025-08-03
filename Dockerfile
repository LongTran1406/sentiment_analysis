FROM python:3.10
WORKDIR /app

COPY requirements.txt requirements.txt
COPY app.py app.py
COPY best_model.pkl best_model.pkl
COPY best_vectorizer.pkl best_vectorizer.pkl 

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]