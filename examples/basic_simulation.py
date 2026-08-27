import sys
sys.path.append('..')

from src.network_simulator import NetworkSimulator
from src.performance_analyzer import PerformanceAnalyzer
from src.visualization import NetworkVisualizer
from src.utils import generate_random_scenario, save_scenario
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility

def main():
    # Generate random network scenario
    print("Creating network scenario...")
    network = generate_random_scenario(num_bs=5, num_users=20)
    
    # Simulate the network
    print("Simulating network...")
    results = network.simulate_network(path_loss_model='CI', bandwidth=100e6)
    
    # Analyze performance
    print("Analyzing performance...")
    analyzer = PerformanceAnalyzer()
    coverage_df, coverage_pct = analyzer.analyze_coverage(network)
    throughput_df = analyzer.throughput_comparison(network)
    path_loss_df = analyzer.analyze_path_loss_models(network)
    
    print(f"\nCoverage: {coverage_pct:.1f}%")
    print(f"Average SNR: {coverage_df['snr'].mean():.1f} dB")
    print(f"Average Capacity: {coverage_df['capacity'].mean()/1e6:.1f} Mbps")
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Network layout
    ax1 = plt.subplot(2, 2, 1)
    NetworkVisualizer.plot_network_layout(network, ax=ax1)
    
    # 2. SNR heatmap
    ax2 = plt.subplot(2, 2, 2)
    NetworkVisualizer.plot_heatmap(network, metric='snr', ax=ax2)
    
    # 3. Throughput comparison
    ax3 = plt.subplot(2, 2, 3)
    NetworkVisualizer.plot_throughput_comparison(throughput_df, ax=ax3)
    
    # 4. Path loss comparison
    ax4 = plt.subplot(2, 2, 4)
    NetworkVisualizer.plot_path_loss_comparison(path_loss_df, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('network_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\nThroughput Statistics:")
    print(throughput_df.to_string(index=False))
    
    # Save scenario for later use
    save_scenario(network, 'scenario.json')
    print("\nScenario saved to scenario.json")

if __name__ == "__main__":
    main()