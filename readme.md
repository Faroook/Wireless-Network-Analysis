# Wireless Network Performance Analysis

A comprehensive Python simulation tool for analyzing wireless network performance metrics including path loss, signal strength, SNR, and Shannon channel capacity with focus on 5G and 6G technologies.

## Features

- **Multiple Path Loss Models**: FSPL, CI, FI, ABG, Ray-tracing
- **Signal Analysis**: RSSI, SNR, Noise Power calculations
- **Capacity Analysis**: Shannon-Hartley, MIMO capacity for 5G/6G
- **Network Simulation**: Multi-cell, multi-user scenarios
- **Performance Metrics**: Coverage, Throughput comparisons
- **Visualization Tools**: Heatmaps, Network layout, Performance plots
- **6G Ready**: Support for THz frequencies, massive MIMO

## Quick Start

```python
from src.network_simulator import NetworkSimulator
from src.performance_analyzer import PerformanceAnalyzer
from src.visualization import NetworkVisualizer
from src.utils import generate_random_scenario

# Create and simulate network
network = generate_random_scenario()
results = network.simulate_network()

# Analyze performance
analyzer = PerformanceAnalyzer()
coverage = analyzer.analyze_coverage(network)
throughput = analyzer.throughput_comparison(network)

# Visualize results
NetworkVisualizer.plot_heatmap(network)
NetworkVisualizer.plot_throughput_comparison(throughput)