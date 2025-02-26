# TIF Backend & Model AI

A comprehensive backend system for traffic monitoring, anomaly detection, and forecasting with integrated AI capabilities.

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Backend Setup](#backend-setup)
3. [APIs](#apis)
   - [Network Traffic Monitoring API](#1-network-traffic-monitoring-api)
   - [Network Traffic Forecasting API](#2-network-traffic-forecasting-api)
   - [Network Error Log Analysis API](#3-network-error-log-analysis-api)
4. [Model Components](#model-components)
5. [Error Handling](#error-handling)
6. [Cache Behavior](#cache-behavior)

## Repository Structure
```
+-- app.py                      # Main backend application
+-- config.py                   # Configuration settings
+-- dockerfile                  # Docker configuration
+-- requirements.txt            # Project dependencies
+-- app_anomalies/             # Anomaly detection module
    +-- __init__.py
    +-- routes.py
+-- app_chatbot/               # Chatbot module
    +-- __init__.py
    +-- routes.py
+-- app_error/                 # Error log module
    +-- __init__.py
    +-- routes.py
+-- app_forecasting/           # Traffic forecasting module
    +-- __init__.py
    +-- routes.py
+-- auth/                      # Authentication module
    +-- __init__.py
    +-- routes.py
+-- cache/                     # Cache storage
+-- data/                      # Data storage
    +-- processed/             # Processed data files
        +-- rrd_log.txt
        +-- rrd_log_minutes.txt
    +-- scripts/               # Database insertion scripts
        +-- insertDB_cacti.py
        +-- insertDB_log.py
        +-- insertDB_weathermap.py
+-- db/                        # Database module
    +-- __init__.py
    +-- routes.py
+-- model/                     # AI models
    +-- anomalies/            # Anomaly detection models
        +-- data_loader.py
        +-- main.py
        +-- models.py
        +-- utils.py
    +-- forecasting/          # Forecasting models
        +-- data_loader.py
        +-- main.py
        +-- models.py
        +-- utils.py
+-- prompt/                    # Prompt templates
    +-- system.py
+-- tools/                     # Utility tools for chatbot
    +-- __init__.py
    +-- error_log.py
    +-- monitoring_anomalies.py
    +-- traffic_forecasting.py
```

## Backend Setup

### Current Status
- **Enabled Routes**: Traffic forecasting
- **Disabled Routes**: Authentication, Chatbot, Anomaly detection, Error handling

### Running the Backend
```bash
python app.py
# Server starts on 0.0.0.0:8080
```

## APIs

### 1. Network Traffic Monitoring API
Base URL: `http://36.67.62.245:8080`

#### 1.1 GET /api/links
Get list of available network links.

**Request:**
```bash
curl -X GET "http://36.67.62.245:8080/api/links"
```

**Response:**
```json
[
    {
        "id": "p-d1-bds_traffic_in_158198",
        "name": "p-d1-bds_traffic_in_158198"
    }
]
```

#### 1.2 GET /api/traffic-data
Get traffic data with time filtering.

**Parameters:**
- `link` (string): Link ID
- `timeFilter` (string): Time range (6h, 1d, 2d)

**Request Examples:**
```bash
# Get all traffic data
curl -X GET "http://36.67.62.245:8080/api/traffic-data?link=&timeFilter=2d"

# Get specific link data
curl -X GET "http://36.67.62.245:8080/api/traffic-data?link=p-d1-bds_traffic_in_158198&timeFilter=2d"

# Get 6-hour data
curl -X GET "http://36.67.62.245:8080/api/traffic-data?link=p-d1-bds_traffic_in_158205&timeFilter=6h"
```

**Response:**
```json
{
    "anomalies": [],
    "dates": ["Sun, 22 Sep 2024 19:10:00 GMT", "Sun, 22 Sep 2024 19:15:00 GMT"],
    "stats": {
        "current_hour": {
            "in": 700.87,
            "out": 912.46
        },
        "peak": {
            "in": 1281.34,
            "out": 1732.42
        }
    },
    "timeRange": {
        "end": "2024-09-24 17:10:00",
        "start": "2024-09-22 19:10:00"
    },
    "trafficIn": [331.54, 341.00],
    "trafficOut": [436.37, 444.66]
}
```

#### 1.3 GET /api/error-logs
Get error logs for specific hostname and timestamp.

**Parameters:**
- `hostname` (string): Link ID/hostname
- `timestamp` (string): Format: YYYY-MM-DD HH:MM:SS

**Request:**
```bash
curl -X GET "http://36.67.62.245:8080/api/error-logs?hostname=p-d1-bds_traffic_in_158203&timestamp=2024-09-23%2017%3A30%3A00"
```

**Response:**
```json
[
    {
        "timestamp": "2024-09-23 17:30:00",
        "device": "ROUTER1",
        "message": "Interface Down",
        "details": "Status change notification"
    }
]
```

### 2. Network Traffic Forecasting API
Base URL: `http://36.67.62.245:8080`

#### 2.1 GET /api/traffic-data (Forecasting)
Get traffic data with forecasts.

**Parameters:**
- `link` (optional): Link ID for specific link forecast

**Request Examples:**
```bash
# Get overview of all links
curl -X GET "http://36.67.62.245:8080/api/traffic-data"

# Get specific link forecast
curl -X GET "http://36.67.62.245:8080/api/traffic-data?link=p-d1-bds_traffic_in_146008"
```

**Response:**
```json
{
    "dates": ["2024-12-01", "2024-12-02", "2024-12-03"],
    "trafficIn": [1.5, 1.8, 1.6],
    "trafficOut": [0.8, 0.9, 0.7],
    "forecastIn": [null, null, null, 1.7, 1.8, 1.9],
    "forecastOut": [null, null, null, 0.8, 0.9, 1.0],
    "allocations": {
        "current_week": {
            "in": 10.5,
            "out": 5.8
        },
        "next_week": {
            "in": 12.0,
            "out": 6.5
        },
        "current_month": {
            "in": 45.2,
            "out": 25.1
        },
        "next_month": {
            "in": 48.5,
            "out": 27.3
        }
    }
}
```

### 3. Network Error Log Analysis API
Base URL: `http://192.168.5.254:5002`

#### 3.1 GET /api/processing_progress
Get real-time progress of log processing.

**Request:**
```bash
curl -N -X GET "http://36.67.62.245:8080/api/processing_progress"
```

**Response (SSE):**
```json
{
    "progress": 45,
    "status": "Processing logs...",
    "detail": "Processed 5,000 of 10,000 logs"
}
```

#### 3.2 GET /api/filters
Get available filter options.

**Request:**
```bash
curl -X GET "http://36.67.62.245:8080/api/filters"
```

**Response:**
```json
{
    "months": ["2024-07", "2024-08", "2024-09"],
    "devices": ["E-D2-CKA", "P-D1-BDS", "P-D2-CKA"],
    "error_types": ["PLATFORM-VIC-4-RFI+", "LINK-3-UPDOWN"]
}
```

#### 3.3 GET /api/top_devices
Get top devices by error count.

**Parameters:**
- `month` (optional): Filter by month (YYYY-MM format)

**Request Examples:**
```bash
# Get overall top devices
curl -X GET "http://36.67.62.245:8080/api/top_devices"

# Get top devices for specific month
curl -X GET "http://36.67.62.245:8080/api/top_devices?month=2024-09"
```

**Response:**
```json
{
    "P-D7-MDO": 150,
    "P-D1-BDS": 120,
    "P-D2-CKA": 85
}
```

#### 3.4 GET /api/ranked_issues
Get ranked list of network issues.

**Request:**
```bash
curl -X GET "http://36.67.62.245:8080/api/ranked_issues"
```

**Response:**
```json
[
    {
        "error_type": "PLATFORM-VIC-4-RFI+",
        "count": 250,
        "most_affected_device": "P-D7-MDO",
        "device_count": 45
    }
]
```

#### 3.5 GET /api/trends
Get error trends data.

**Parameters:**
- `month` (optional): Filter by month (YYYY-MM format)
- `device` (optional): Filter by device name

**Request Examples:**
```bash
# Get overall trends
curl -X GET "http://36.67.62.245:8080/api/trends?month=&device="

# Get trends for specific month and device
curl -X GET "http://36.67.62.245:8080/api/trends?month=2024-09&device=P-D7-MDO"
```

**Response:**
```json
[
  {
    "L2-BFD-6-SESSION_DAMPENING_OFF": 14,
    "L2-BFD-6-SESSION_DAMPENING_ON": 14,
    "L2-BFD-6-SESSION_NO_RESOURCES": 2,
    "L2-BFD-6-SESSION_REMOVED": 13,
    "L2-BFD-6-SESSION_STATE_DOWN": 1,
    "L2-BFD-6-SESSION_STATE_UP": 14,
    "date": "2024-07-01"
  }
]
```

#### 3.6 GET /api/detailed_logs
Get detailed log entries.

**Parameters:**
- `page` (number): Page number (default: 1)
- `per_page` (number): Items per page (10, 25, 50, 100)
- `issue_type` (string): Filter by error type
- `severity` (string): Filter by severity (CRITICAL, WARNING, INFO)
- `search` (string): Search in log details

**Request:**
```bash
curl -X GET "http://36.67.62.245:8080/api/detailed_logs?page=1&per_page=100&issue_type=PLATFORM-VIC-4-RFI+&severity=CRITICAL&search="
```

**Response:**
```json
{
    "logs": [
        {
            "timestamp": "2024-09-23T17:30:00",
            "device": "P-D7-MDO",
            "error_type": "PLATFORM-VIC-4-RFI+",
            "severity": "CRITICAL",
            "interface": "GigabitEthernet1/0/1",
            "status": "down",
            "details": "Interface GigabitEthernet1/0/1 changed state to down"
        }
    ],
    "total": 150
}
```

## Example API Flows

### Traffic Monitoring Flow
```plaintext
1. Initial Load:
   GET /api/links
   GET /api/traffic-data?link=&timeFilter=2d

2. View Specific Link:
   GET /api/traffic-data?link=p-d1-bds_traffic_in_158198&timeFilter=2d
   GET /api/traffic-data?link=p-d1-bds_traffic_in_158198&timeFilter=1d

3. Change Time Filter:
   GET /api/traffic-data?link=p-d1-bds_traffic_in_158205&timeFilter=1d
   GET /api/traffic-data?link=p-d1-bds_traffic_in_158205&timeFilter=6h

4. View Error Logs:
   GET /api/error-logs?hostname=p-d1-bds_traffic_in_158203&timestamp=2024-09-23%2017%3A30%3A00
```

### Traffic Forecasting Flow
```plaintext
1. Initial Page Load:
   GET /api/links
   GET /api/traffic-data

2. View Individual Link Forecasts:
   GET /api/traffic-data?link=p-d1-bds_traffic_in_146008
   GET /api/traffic-data?link=p-d1-bds_traffic_in_146012
```

### Error Analysis Flow
```plaintext
1. Initial Load:
   GET /api/processing_progress
   GET /api/filters
   GET /api/ranked_issues
   GET /api/top_devices?month=
   GET /api/trends?month=&device=

2. View Monthly Data:
   GET /api/top_devices?month=2024-09
   GET /api/trends?month=2024-09&device=

3. View Device Specific Data:
   GET /api/trends?month=2024-09&device=P-D7-MDO

4. View Detailed Logs:
   GET /api/detailed_logs?page=1&per_page=100&issue_type=PLATFORM-VIC-4-RFI+&severity=CRITICAL&search=
```

## Model Components

### Anomaly Detection Features
- Multiple detection algorithms
- Automated model evaluation
- Performance visualization
- Model checkpointing

### Running Model Pipelines
```bash
python model/anomalies/main.py
python model/forecasting/main.py
```

## Error Handling
Status Codes:
- 200: Success
- 404: Data not found
- 500: Server error

Error Response Format:
```json
{
    "error": "Error message description"
}
```

## Cache Behavior
- Traffic data cached for 24 hours
- Initial forecast requests may take longer
- Real-time processing updates via SSE
- Filter data cached for improved performance
