import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from .channel_models import (
    PathLossModels, SignalStrengthCalculator, 
    SNRCalculator, ShannonCapacity
)

@dataclass
class BaseStation:
    """Base station configuration"""
    id: int
    x: float
    y: float
    frequency: float  # GHz
    tx_power: float   # dBm
    technology: str = '5G'  # 5G, 6G, etc.
    
@dataclass
class UserEquipment:
    """User equipment configuration"""
    id: int
    x: float
    y: float
    velocity: Tuple[float, float] = (0, 0)
    
class NetworkSimulator:
    """Main network simulation engine"""
    
    def __init__(self, area_size: Tuple[float, float] = (1000, 1000)):
        self.area_size = area_size
        self.base_stations: List[BaseStation] = []
        self.user_equipment: List[UserEquipment] = []
        self.channel_data: Dict = {}
        
    def add_base_station(self, x: float, y: float, frequency: float,
                        tx_power: float, technology: str = '5G'):
        """Add a base station to the network"""
        bs_id = len(self.base_stations)
        bs = BaseStation(bs_id, x, y, frequency, tx_power, technology)
        self.base_stations.append(bs)
        return bs
    
    def add_user(self, x: float, y: float, velocity: Tuple[float, float] = (0, 0)):
        """Add a user equipment to the network"""
        ue_id = len(self.user_equipment)
        ue = UserEquipment(ue_id, x, y, velocity)
        self.user_equipment.append(ue)
        return ue
    
    def calculate_distance(self, bs: BaseStation, ue: UserEquipment) -> float:
        """Calculate Euclidean distance between BS and UE"""
        return np.sqrt((bs.x - ue.x)**2 + (bs.y - ue.y)**2)
    
    def simulate_connection(self, bs: BaseStation, ue: UserEquipment,
                           path_loss_model: str = 'CI_path_loss',
                           bandwidth: float = 100e6) -> Dict:
        """Simulate connection between a base station and user"""
        distance = self.calculate_distance(bs, ue)
        
        # Get the path loss function
        path_loss_func = getattr(PathLossModels, path_loss_model, PathLossModels.CI_path_loss)
        
        # Calculate path loss
        path_loss = path_loss_func(distance, bs.frequency)
        
        # Received signal strength
        rx_power = SignalStrengthCalculator.calculate_rx_power(
            bs.tx_power, path_loss
        )
        
        # Noise power and SNR
        noise_power = SNRCalculator.calculate_noise_power(bandwidth)
        snr = SNRCalculator.calculate_snr(rx_power, noise_power)
        
        # Channel capacity - FIXED: Changed ShannonChannelCapacity to ShannonCapacity
        capacity = ShannonCapacity.channel_capacity(bandwidth, snr)
        
        # MIMO capacity (assuming 4x4 MIMO for 5G, 64x64 for 6G)
        if bs.technology == '6G':
            mimo_capacity = ShannonCapacity.achievable_rate_with_mimo(
                bandwidth, snr, 64
            )
        else:
            mimo_capacity = ShannonCapacity.achievable_rate_with_mimo(
                bandwidth, snr, 4
            )
        
        return {
            'distance': distance,
            'path_loss': path_loss,
            'rx_power': rx_power,
            'noise_power': noise_power,
            'snr': snr,
            'capacity': capacity,
            'mimo_capacity': mimo_capacity,
            'spectral_efficiency': ShannonCapacity.spectral_efficiency(snr)
        }
    
    def simulate_network(self, path_loss_model: str = 'CI',
                        bandwidth: float = 100e6) -> pd.DataFrame:
        """Simulate entire network"""
        results = []
        
        # Map user-friendly names to actual method names
        model_mapping = {
            'FSPL': 'FSPL_path_loss',
            'CI': 'CI_path_loss',
            'FI': 'FI_path_loss',
            'ABG': 'ABG_path_loss',
            'RayTracing': 'RayTracing_path_loss'
        }
        
        # Get the actual method name
        actual_model = model_mapping.get(path_loss_model, 'CI_path_loss')
        
        for ue in self.user_equipment:
            for bs in self.base_stations:
                result = self.simulate_connection(
                    bs, ue, actual_model, bandwidth
                )
                result.update({
                    'bs_id': bs.id,
                    'ue_id': ue.id,
                    'technology': bs.technology,
                    'bs_x': bs.x,
                    'bs_y': bs.y,
                    'ue_x': ue.x,
                    'ue_y': ue.y
                })
                results.append(result)
        
        return pd.DataFrame(results)
    
    def find_best_connection(self, ue: UserEquipment) -> Dict:
        """Find the best base station for a user"""
        best_results = []
        
        for bs in self.base_stations:
            result = self.simulate_connection(bs, ue)
            best_results.append({
                'bs_id': bs.id,
                'snr': result['snr'],
                'capacity': result['capacity'],
                'distance': result['distance']
            })
        
        return max(best_results, key=lambda x: x['snr'])