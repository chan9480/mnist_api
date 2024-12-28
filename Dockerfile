# Use the official Python 3.10 slim image as the base image
FROM python:3.10-slim

# Set environment variables to avoid writing .pyc files and to ensure stdout/stderr are not buffered
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file first to leverage caching
COPY requirements.txt /app/

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# COPY . /app/

# Expose the port the Flask app will run on (Flask default is 5000)
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]