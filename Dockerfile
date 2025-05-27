# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=dapp.py
ENV FLASK_RUN_HOST=0.0.0.0

# Set the working directory in the container
WORKDIR /app

# Copy your custom requirements file
COPY reqdock.txt /app/reqdock.txt

# Install dependencies from reqdock.txt
RUN pip install --no-cache-dir -r reqdock.txt

# Copy your Flask app
COPY dapp.py /app/

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask app
CMD ["flask", "run"]
