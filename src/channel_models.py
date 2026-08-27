import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class ChannelParameters:
    """Parameters for different channel models"""
    frequency: float  # in GHz
    tx_power: float   # in dBm
    antenna_gain_tx: float = 0  # dBi
    antenna_gain_rx: float = 0  # dBi
    system_loss: float = 0  # dB
    
class PathLossModels:
    """Implementation of various path loss models"""
    
    @staticmethod
    def free_space_path_loss(distance: float, frequency: float) -> float:
        """Free Space Path Loss model"""
        # FSPL = 20*log10(d) + 20*log10(f) + 32.44
        return 20 * np.log10(distance) + 20 * np.log10(frequency) + 32.44
    
    @staticmethod
    def close_in_free_space(distance: float, frequency: float, 
                           reference_distance: float = 1.0) -> float:
        """Close-in (CI) Free Space Reference Distance model"""
        # PL_CI = FSPL(f, d0) + 10*n*log10(d/d0) + X_sigma
        n = 2.0  # Path loss exponent
        fspl = PathLossModels.free_space_path_loss(reference_distance, frequency)
        return fspl + 10 * n * np.log10(distance / reference_distance)
    
    @staticmethod
    def floating_intercept(distance: float, frequency: float, 
                          alpha: float = 2.5, beta: float = 2.8) -> float:
        """Floating Intercept (FI) model for 5G/6G"""
        # PL_FI = alpha*10*log10(d) + beta*10*log10(f) + X_sigma
        return alpha * 10 * np.log10(distance) + beta * 10 * np.log10(frequency)
    
    @staticmethod
    def alpha_beta_gamma(distance: float, frequency: float, 
                         alpha: float = 3.5, beta: float = 2.0, 
                         gamma: float = 1.5) -> float:
        """Alpha-Beta-Gamma (ABG) model for mmWave/THz"""
        # PL_ABG = alpha*10*log10(d) + beta*10*log10(f) + gamma
        return alpha * 10 * np.log10(distance) + beta * 10 * np.log10(frequency) + gamma
    
    @staticmethod
    def ray_tracing_path_loss(distance: float, frequency: float,
                             environment: str = 'urban') -> float:
        """Simplified ray-tracing model"""
        # Different path loss exponents for different environments
        exponents = {
            'urban': 3.5,
            'suburban': 3.0,
            'rural': 2.7,
            'indoor': 2.5
        }
        n = exponents.get(environment, 3.0)
        return PathLossModels.close_in_free_space(distance, frequency) + 10*n*np.log10(distance)
    
    # Wrapper methods for the naming convention expected by simulate_connection
    @staticmethod
    def FSPL_path_loss(distance: float, frequency: float) -> float:
        """Wrapper for FSPL"""
        return PathLossModels.free_space_path_loss(distance, frequency)
    
    @staticmethod
    def CI_path_loss(distance: float, frequency: float) -> float:
        """Wrapper for CI"""
        return PathLossModels.close_in_free_space(distance, frequency)
    
    @staticmethod
    def FI_path_loss(distance: float, frequency: float) -> float:
        """Wrapper for FI"""
        return PathLossModels.floating_intercept(distance, frequency)
    
    @staticmethod
    def ABG_path_loss(distance: float, frequency: float) -> float:
        """Wrapper for ABG"""
        return PathLossModels.alpha_beta_gamma(distance, frequency)
    
    @staticmethod
    def RayTracing_path_loss(distance: float, frequency: float) -> float:
        """Wrapper for RayTracing"""
        return PathLossModels.ray_tracing_path_loss(distance, frequency)

class SignalStrengthCalculator:
    """Calculate received signal strength and related metrics"""
    
    @staticmethod
    def calculate_rx_power(tx_power: float, path_loss: float, 
                          tx_gain: float = 0, rx_gain: float = 0,
                          system_loss: float = 0) -> float:
        """Calculate received power in dBm"""
        return tx_power + tx_gain + rx_gain - path_loss - system_loss
    
    @staticmethod
    def calculate_rssi_from_distance(distance: float, frequency: float,
                                     tx_power: float, model: str = 'FSPL') -> float:
        """Calculate RSSI for a given distance"""
        path_loss_models = {
            'FSPL': PathLossModels.free_space_path_loss,
            'CI': PathLossModels.close_in_free_space,
            'FI': PathLossModels.floating_intercept,
            'ABG': PathLossModels.alpha_beta_gamma
        }
        
        if model in path_loss_models:
            pl = path_loss_models[model](distance, frequency)
        else:
            pl = PathLossModels.free_space_path_loss(distance, frequency)
            
        return SignalStrengthCalculator.calculate_rx_power(tx_power, pl)

class SNRCalculator:
    """Signal-to-Noise Ratio calculations"""
    
    @staticmethod
    def calculate_snr(rx_power: float, noise_power: float) -> float:
        """Calculate SNR in dB"""
        return rx_power - noise_power
    
    @staticmethod
    def calculate_noise_power(bandwidth: float, temperature: float = 290,
                             noise_figure: float = 5) -> float:
        """Calculate thermal noise power in dBm"""
        # N = kTB + NF
        k = 1.38e-23  # Boltzmann constant
        noise_linear = k * temperature * bandwidth * 1000  # Convert to mW
        noise_dbm = 10 * np.log10(noise_linear)
        return noise_dbm + noise_figure  # Add noise figure in dB
    
    @staticmethod
    def calculate_snr_from_distance(distance: float, frequency: float,
                                   tx_power: float, bandwidth: float,
                                   model: str = 'FSPL') -> float:
        """Calculate SNR directly from distance"""
        rssi = SignalStrengthCalculator.calculate_rssi_from_distance(
            distance, frequency, tx_power, model
        )
        noise = SNRCalculator.calculate_noise_power(bandwidth)
        return SNRCalculator.calculate_snr(rssi, noise)

class ShannonCapacity:
    """Shannon-Hartley channel capacity calculations"""
    
    @staticmethod
    def channel_capacity(bandwidth: float, snr: float) -> float:
        """Calculate channel capacity in bits/second"""
        # C = B * log2(1 + SNR)
        snr_linear = 10 ** (snr / 10)
        return bandwidth * np.log2(1 + snr_linear)
    
    @staticmethod
    def spectral_efficiency(snr: float) -> float:
        """Calculate spectral efficiency in bits/s/Hz"""
        snr_linear = 10 ** (snr / 10)
        return np.log2(1 + snr_linear)
    
    @staticmethod
    def achievable_rate_with_mimo(bandwidth: float, snr: float, 
                                 num_antennas: int) -> float:
        """Capacity with MIMO systems (ideal conditions)"""
        snr_linear = 10 ** (snr / 10)
        return bandwidth * num_antennas * np.log2(1 + snr_linear / num_antennas)