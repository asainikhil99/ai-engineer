FROM python:3.13-slim

WORKDIR /app

COPY req_API.txt .
RUN pip install -r req_API.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "fastAPI_basics:app", "--host", "0.0.0.0", "--port", "8000"]