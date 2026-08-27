import json
import numpy as np
from typing import Dict, Any
from .network_simulator import NetworkSimulator

def save_scenario(network: NetworkSimulator, filename: str):
    """Save network configuration to JSON file"""
    scenario = {
        'area_size': network.area_size,
        'base_stations': [
            {
                'id': bs.id,
                'x': bs.x,
                'y': bs.y,
                'frequency': bs.frequency,
                'tx_power': bs.tx_power,
                'technology': bs.technology
            }
            for bs in network.base_stations
        ],
        'users': [
            {
                'id': ue.id,
                'x': ue.x,
                'y': ue.y,
                'velocity': ue.velocity
            }
            for ue in network.user_equipment
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(scenario, f, indent=2)

def load_scenario(filename: str) -> NetworkSimulator:
    """Load network configuration from JSON file"""
    with open(filename, 'r') as f:
        scenario = json.load(f)
    
    network = NetworkSimulator(tuple(scenario['area_size']))
    
    for bs_data in scenario['base_stations']:
        network.add_base_station(
            bs_data['x'], bs_data['y'],
            bs_data['frequency'], bs_data['tx_power'],
            bs_data['technology']
        )
    
    for ue_data in scenario['users']:
        network.add_user(
            ue_data['x'], ue_data['y'],
            tuple(ue_data['velocity'])
        )
    
    return network

def generate_random_scenario(num_bs: int = 5, num_users: int = 20,
                           area_size: tuple = (1000, 1000)) -> NetworkSimulator:
    """Generate random network scenario"""
    network = NetworkSimulator(area_size)
    
    # Add base stations
    technologies = ['5G', '6G']
    for i in range(num_bs):
        x = np.random.uniform(0, area_size[0])
        y = np.random.uniform(0, area_size[1])
        freq = np.random.choice([28, 39, 73, 140])  # GHz
        tx_power = np.random.uniform(20, 46)  # dBm
        tech = np.random.choice(technologies)
        network.add_base_station(x, y, freq, tx_power, tech)
    
    # Add users
    for i in range(num_users):
        x = np.random.uniform(0, area_size[0])
        y = np.random.uniform(0, area_size[1])
        network.add_user(x, y)
    
    return network