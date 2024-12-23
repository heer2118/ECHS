# Use a lightweight base image with Python
FROM nvcr.io/nvidia/pytorch:22.12-py3

# Set the working directory
WORKDIR /app

# Copy application code to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask app port
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask app
CMD ["flask", "run"]
