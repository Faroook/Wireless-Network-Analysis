import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .network_simulator import NetworkSimulator, BaseStation, UserEquipment
from .channel_models import PathLossModels, SignalStrengthCalculator, ShannonCapacity

class PerformanceAnalyzer:
    """Analyze network performance metrics"""
    
    @staticmethod
    def analyze_coverage(network: NetworkSimulator, 
                        snr_threshold: float = 10) -> Tuple[pd.DataFrame, float]:
        """Analyze coverage based on SNR threshold"""
        results = []
        
        for ue in network.user_equipment:
            best_connection = network.find_best_connection(ue)
            results.append({
                'ue_id': ue.id,
                'best_bs': best_connection['bs_id'],
                'snr': best_connection['snr'],
                'capacity': best_connection['capacity'],
                'covered': best_connection['snr'] >= snr_threshold
            })
        
        df = pd.DataFrame(results)
        coverage_percentage = (df['covered'].sum() / len(df)) * 100
        
        return df, coverage_percentage
    
    @staticmethod
    def throughput_comparison(network: NetworkSimulator) -> pd.DataFrame:
        """Compare throughput across different technologies"""
        technologies = ['5G', '6G']
        results = []
        
        for tech in technologies:
            tech_results = []
            for ue in network.user_equipment:
                for bs in network.base_stations:
                    if bs.technology == tech:
                        result = network.simulate_connection(bs, ue)
                        tech_results.append(result['mimo_capacity'])
            
            if tech_results:
                results.append({
                    'technology': tech,
                    'mean_throughput': np.mean(tech_results) / 1e6,  # Mbps
                    'max_throughput': np.max(tech_results) / 1e6,
                    'min_throughput': np.min(tech_results) / 1e6,
                    'std_throughput': np.std(tech_results) / 1e6
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def analyze_path_loss_models(network: NetworkSimulator) -> pd.DataFrame:
        """Compare different path loss models"""
        models = ['FSPL', 'CI', 'FI', 'ABG', 'RayTracing']
        results = []
        
        # Get the first user and first base station for comparison
        if not network.user_equipment or not network.base_stations:
            return pd.DataFrame()
        
        ue = network.user_equipment[0]
        bs = network.base_stations[0]
        distance = network.calculate_distance(bs, ue)
        
        # Test at different distances
        test_distances = np.linspace(1, 500, 20)
        
        for dist in test_distances:
            for model in models:
                try:
                    # Map model name to method
                    model_methods = {
                        'FSPL': PathLossModels.free_space_path_loss,
                        'CI': PathLossModels.close_in_free_space,
                        'FI': PathLossModels.floating_intercept,
                        'ABG': PathLossModels.alpha_beta_gamma,
                        'RayTracing': PathLossModels.ray_tracing_path_loss
                    }
                    
                    path_loss_func = model_methods.get(model)
                    if path_loss_func:
                        path_loss = path_loss_func(dist, bs.frequency)
                        rx_power = SignalStrengthCalculator.calculate_rx_power(
                            bs.tx_power, path_loss
                        )
                        results.append({
                            'model': model,
                            'distance': dist,
                            'path_loss': path_loss,
                            'rx_power': rx_power,
                            'frequency': bs.frequency
                        })
                except Exception as e:
                    continue
        
        return pd.DataFrame(results)