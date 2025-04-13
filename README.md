# Smart Health Monitoring System

A comprehensive health monitoring system that collects, analyzes, and displays vital health data in real-time, providing risk assessments and alerts for potential health issues.

## Project Components

- **Smart_Health**: Front-end web application
- **CloudProcessor**: Backend service for data processing and analysis
- **Simulator**: Tool to generate simulated health data

## Prerequisites

- Node.js and npm
- Python 3.6+
- SQLite (included in Python)

## Setup Instructions

### 1. Front-end Setup (Smart_Health)

```bash
cd Smart_Health
npm install
```

### 2. CloudProcessor Setup

```bash
cd CloudProcessor
pip install flask flask-cors numpy pandas scikit-learn threading
```

### 3. Simulator Setup

```bash
cd Simulator
pip install requests numpy random time
```

### 4. Initialize Database

Run the database initialization script to create the database and populate it with initial data:

```bash
cd CloudProcessor
python db_script.py
```

### 5. Train the Machine Learning Model

Before running the application, you need to train the machine learning model:

```bash
cd CloudProcessor
python train.py
```

## Running the Application

### 1. Start the Backend Service

```bash
cd CloudProcessor
python Cloud.py
```

### 2. Start the Front-end Application

In a new terminal:

```bash
cd Smart_Health
npm run dev
```

### 3. Run the Simulator

In another terminal:

```bash
cd Simulator
python main.py
```

## Login Credentials

Access the web interface at the URL displayed in the Smart_Health terminal (typically http://localhost:5173/)

Use the following credentials to log in:
- **Username**: demo
- **Password**: password

## Features

- Real-time health monitoring
- Historical health data trends
- Risk assessment and prediction
- Anomaly detection and alerts
- Dashboard visualization of vital signs

## Troubleshooting

- Ensure all components are running simultaneously
- Check that ports 5000 (backend) and 5173 (frontend) are available
- Verify database initialization was successful
- Check console logs for any error messages
