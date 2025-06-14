FROM python:3.9-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app


COPY re.txt /app/
COPY app.py /app/
COPY templates /app/templates
COPY static /app/static


RUN pip install --no-cache-dir -r re.txt

EXPOSE 5000


CMD ["python", "app.py"]