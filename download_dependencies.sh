#!/bin/bash
echo "Downloading monitoring dependencies..."

# Download Prometheus
if [ ! -f "prometheus-2.47.0.linux-amd64.tar.gz" ]; then
    wget https://github.com/prometheus/prometheus/releases/download/v2.47.0/prometheus-2.47.0.linux-amd64.tar.gz
    tar xvf prometheus-2.47.0.linux-amd64.tar.gz
fi

# Download Loki
if [ ! -f "loki-linux-amd64.zip" ]; then
    wget https://github.com/grafana/loki/releases/download/v2.9.0/loki-linux-amd64.zip
    unzip loki-linux-amd64.zip
fi

# Download Promtail  
if [ ! -f "promtail-linux-amd64.zip" ]; then
    wget https://github.com/grafana/loki/releases/download/v2.9.0/promtail-linux-amd64.zip
    unzip promtail-linux-amd64.zip
fi

echo "Dependencies downloaded!"
