# AI-Driven Predictive Maintenance Dashboard

This repository hosts an AI-Driven Predictive Maintenance Dashboard, a cloud-native application developed for the CNCF Hackathon. It monitors compressor performance, predicts potential failures using machine learning, and provides actionable maintenance insights through an interactive dashboard. Built with Dash, Flask, and Plotly, it integrates with Prometheus and Grafana for real-time metrics monitoring and Loki for log aggregation, making it ideal for industrial IoT and predictive maintenance use cases.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Monitoring Setup](#monitoring-setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Real-Time Monitoring**: Tracks compressor metrics (pressure, temperature, vibration, current, efficiency).
- **Predictive Analytics**: Uses AI to predict compressor failures and prioritize maintenance.
- **Interactive Dashboard**: Visualizes sensor data, trends, 3D plots, and failure probabilities using Dash and Plotly.
- **Fault Injection**: Simulates faults (e.g., high vibration, overheating) for testing.
- **Prometheus Metrics**: Exposes custom metrics for compressor health, alerts, and app performance.
- **Grafana Integration**: Visualizes metrics and logs with Grafana and Loki.
- **Secure Access**: Implements Flask-based authentication for dashboard access.

## Architecture

The application uses a cloud-native architecture:

- **Frontend**: Dash with Bootstrap for a responsive UI.
- **Backend**: Flask server with Prometheus metrics exporter.
- **Data Processing**: Simulated sensor data (`data_simulator2.py`) with fault prediction and efficiency calculations (`utils.py`).
- **Monitoring**: Prometheus for metrics, Grafana for visualization, Loki for logs, and Node Exporter for system metrics.
- **Deployment**: Docker Compose for local monitoring setup.

## Prerequisites

- **Python (3.8+)**: For running the application.
- **Docker and Docker Compose**: For monitoring services.
- **Git**: To clone the repository.
- **A modern web browser** (e.g., Chrome, Firefox) for accessing the dashboard and Grafana.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ShreyaDhurde/CNCF-Hackathon-Winners.git
cd CNCF-Hackathon-Winners
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Ensure `requirements.txt` includes dependencies like `dash`, `dash-bootstrap-components`, `plotly`, `pandas`, `flask`, `flask-login`, `prometheus-flask-exporter`, etc.

### 4. Run the Application

```bash
python app2.py
```

Access the dashboard at http://0.0.0.0:8050. Log in with:
- **Username**: `admin`
- **Password**: `admin123`

## Monitoring Setup

Set up Prometheus, Grafana, Loki, and Node Exporter to monitor metrics and logs.

### 1. Create Prometheus Configuration

Create a `prometheus.yml` file in the project root:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  - job_name: 'grafana'
    static_configs:
      - targets: ['grafana:3000']
  - job_name: 'flask_dash_app'
    static_configs:
      - targets: ['<your-local-ip>:8050']  # Replace with your IP (e.g., 10.211.223.102:8050)
```

**Find your local IP:**
- **macOS**: `ifconfig` or `ipconfig getifaddr en0`
- **Linux**: `ip addr` or `hostname -I`
- **Windows**: `ipconfig`

### 2. Create Docker Compose File

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    networks:
      - monitoring

  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge
```

### 3. Start Monitoring Services

```bash
docker-compose up -d
```

Access services:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (Login: admin/admin)
- **Loki**: http://localhost:3100
- **Node Exporter**: http://localhost:9100

### 4. Configure Grafana

1. **Log in to Grafana** (http://localhost:3000).
2. **Add Prometheus data source**:
   - URL: `http://prometheus:9090`
   - Save and test.
3. **Add Loki data source**:
   - URL: `http://loki:3100`
   - Save and test.
4. **Create a dashboard**:
   - Add panels for metrics like `compressor_failure_probability`, `compressor_sensor_reading`, `compressor_efficiency`, `maintenance_queue_size`.
   - Example query: `compressor_failure_probability{compressor_name=~".*"}`
   - Add a log panel for Loki to view `logs/dash_app.log`.

### 5. Access Metrics

View application metrics at http://\<your-ip\>:8050/metrics. Metrics include:

- `compressor_sensor_reading`: Pressure, temperature, vibration, etc.
- `compressor_failure_probability`: Predicted failure likelihood.
- `compressor_maintenance_alerts_total`: Maintenance alerts by severity.
- `sensor_data_processing_seconds`: Data processing time.

## Usage

1. **Dashboard Access**: Navigate to http://0.0.0.0:8050 and log in (admin/admin123).
2. **Monitor Compressors**: Select a compressor to view real-time sensor data, trends, and fault probabilities.
3. **Inject Faults**: Use the fault injection dropdowns to simulate issues (e.g., high vibration) for testing.
4. **View Alerts**: Check "Service Engineer Alerts" for maintenance notifications.
5. **Monitor Metrics**: Use Grafana to visualize compressor and application metrics; view logs via Loki.
6. **Logs**: Application logs are saved to `logs/dash_app.log`.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request.

Adhere to the [CNCF Code of Conduct](https://github.com/cncf/foundation/blob/master/code-of-conduct.md).

## License

[MIT License](LICENSE)

## Acknowledgments

- **CNCF**: For hosting CloudNativeHacks.
- **Dash, Plotly, Flask**: For UI and backend frameworks.
- **Prometheus, Grafana, Loki**: For monitoring and logging.
- **CNCF Hackathon team** for collaboration.

---

For issues or questions, open a GitHub issue or contact the CNCF community.