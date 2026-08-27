import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .network_simulator import NetworkSimulator, BaseStation, UserEquipment
from .channel_models import PathLossModels, SignalStrengthCalculator

class NetworkVisualizer:
    """Visualize network simulations"""
    
    @staticmethod
    def plot_network_layout(network: NetworkSimulator, ax=None):
        """Plot base stations and users"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot base stations
        bs_colors = {'5G': 'red', '6G': 'blue'}
        for bs in network.base_stations:
            color = bs_colors.get(bs.technology, 'green')
            ax.scatter(bs.x, bs.y, s=200, c=color, marker='s',
                      label=f'BS {bs.id} ({bs.technology})')
            ax.annotate(f'BS{bs.id}', (bs.x, bs.y), xytext=(5, 5),
                       textcoords='offset points')
        
        # Plot users
        for ue in network.user_equipment:
            ax.scatter(ue.x, ue.y, s=50, c='black', marker='o', alpha=0.5)
            ax.annotate(f'UE{ue.id}', (ue.x, ue.y), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('X coordinate (m)')
        ax.set_ylabel('Y coordinate (m)')
        ax.set_title('Network Layout')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    @staticmethod
    def plot_heatmap(network: NetworkSimulator, metric: str = 'snr', 
                    resolution: int = 50, ax=None):
        """Create heatmap of network metrics"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        x = np.linspace(0, network.area_size[0], resolution)
        y = np.linspace(0, network.area_size[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Find best metric value for each point
        for i in range(resolution):
            for j in range(resolution):
                # Create temporary user at this position
                temp_ue = UserEquipment(-1, X[i,j], Y[i,j])  # Fixed: UserEquipment is now imported
                best = network.find_best_connection(temp_ue)
                
                if metric == 'snr':
                    Z[i,j] = best['snr']
                elif metric == 'capacity':
                    Z[i,j] = best['capacity'] / 1e6  # Convert to Mbps
                elif metric == 'distance':
                    Z[i,j] = best['distance']
        
        im = ax.imshow(Z, extent=[0, network.area_size[0], 0, network.area_size[1]],
                      origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax, label=f'{metric.upper()}')
        
        # Add base stations
        for bs in network.base_stations:
            ax.scatter(bs.x, bs.y, s=100, c='red', marker='s', 
                      label=f'BS{bs.id}' if bs.id == 0 else "")
        
        ax.set_xlabel('X coordinate (m)')
        ax.set_ylabel('Y coordinate (m)')
        ax.set_title(f'Network {metric.upper()} Coverage Map')
        ax.legend()
        
        return ax
    
    @staticmethod
    def plot_throughput_comparison(throughput_df, ax=None):
        """Plot throughput comparison"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(throughput_df))
        width = 0.35
        
        bars = ax.bar(x, throughput_df['mean_throughput'], width,
                     yerr=throughput_df['std_throughput'],
                     capsize=5, label='Mean Throughput')
        
        ax.set_xlabel('Technology')
        ax.set_ylabel('Throughput (Mbps)')
        ax.set_title('Throughput Comparison: 5G vs 6G')
        ax.set_xticks(x)
        ax.set_xticklabels(throughput_df['technology'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f} Mbps', ha='center', va='bottom')
        
        return ax
    
    @staticmethod
    def plot_path_loss_comparison(path_loss_df, ax=None):
        """Plot path loss model comparison"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        models = path_loss_df['model'].unique()
        
        for model in models:
            model_data = path_loss_df[path_loss_df['model'] == model]
            # Sort by distance for proper line plotting
            model_data = model_data.sort_values('distance')
            ax.plot(model_data['distance'], model_data['path_loss'],
                   label=model, linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Path Loss (dB)')
        ax.set_title('Path Loss Model Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_snr_vs_distance(network: NetworkSimulator, ax=None):
        """Plot SNR vs distance for different technologies"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect data
        distances = []
        snrs = []
        technologies = []
        
        for bs in network.base_stations:
            for ue in network.user_equipment:
                distance = network.calculate_distance(bs, ue)
                result = network.simulate_connection(bs, ue)
                distances.append(distance)
                snrs.append(result['snr'])
                technologies.append(bs.technology)
        
        # Create scatter plot
        colors = {'5G': 'red', '6G': 'blue'}
        for tech in set(technologies):
            tech_indices = [i for i, t in enumerate(technologies) if t == tech]
            tech_distances = [distances[i] for i in tech_indices]
            tech_snrs = [snrs[i] for i in tech_indices]
            ax.scatter(tech_distances, tech_snrs, c=colors[tech], 
                      label=tech, alpha=0.6, s=50)
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('SNR (dB)')
        ax.set_title('SNR vs Distance by Technology')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    @staticmethod
    def create_interactive_dashboard(network: NetworkSimulator):
        """Create interactive plotly dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Network Layout', 'SNR Coverage',
                          'Throughput Analysis', 'Path Loss Analysis')
        )
        
        # Add network layout
        for bs in network.base_stations:
            fig.add_trace(
                go.Scatter(x=[bs.x], y=[bs.y], mode='markers+text',
                          marker=dict(size=15, symbol='square'),
                          text=f'BS{bs.id}', textposition='top center',
                          name=f'BS {bs.technology}'),
                row=1, col=1
            )
        
        # Add users
        for ue in network.user_equipment:
            fig.add_trace(
                go.Scatter(x=[ue.x], y=[ue.y], mode='markers',
                          marker=dict(size=8, symbol='circle'),
                          name=f'UE{ue.id}'),
                row=1, col=1
            )
        
        # Add SNR heatmap (simplified for plotly)
        x = np.linspace(0, network.area_size[0], 30)
        y = np.linspace(0, network.area_size[1], 30)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(30):
            for j in range(30):
                temp_ue = UserEquipment(-1, X[i,j], Y[i,j])
                best = network.find_best_connection(temp_ue)
                Z[i,j] = best['snr']
        
        fig.add_trace(
            go.Heatmap(z=Z, x=x, y=y, colorscale='Viridis', showscale=False),
            row=1, col=2
        )
        
        fig.update_layout(height=800, width=1200,
                         title_text="Wireless Network Performance Dashboard")
        
        return fig