FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    xgboost==2.0.2 \
    scikit-learn==1.3.2 \
    pandas==2.1.3 \
    numpy==1.26.2 \
    joblib==1.3.2 \
    pydantic==2.5.2

COPY main.py .
COPY model.pkl .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
