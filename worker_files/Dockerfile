# Start from official Python 3.13 slim image
FROM python:3.13-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (Pipenv needs gcc, build-essential, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*


# Instale pacotes necessários para acessar o MS-SQLServer
RUN apt-get update && \
    apt-get install --no-install-recommends -y unixodbc unixodbc-dev curl gnupg2 && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install --no-install-recommends -y msodbcsql17 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install pipenv
RUN pip install --upgrade pip pipenv

# Install Gunicorn
RUN pip install gunicorn==23.0.0

# Set work directory
WORKDIR /app

# Copy Pipenv files first for better cache utilization
COPY Pipfile Pipfile.lock ./

# Install dependencies via pipenv
RUN pipenv install --deploy --system

# Copy the rest of the application code
COPY . .

CMD ["python", "main.py"]