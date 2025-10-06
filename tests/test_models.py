"""
Tests for model implementations
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import RNNModel, LSTMModel, GRUModel, TransformerModel


class TestRNNModel:
    """Test RNN model implementation"""
    
    def test_rnn_initialization(self):
        """Test RNN model initialization"""
        model = RNNModel(input_dim=10, hidden_dim=32, output_dim=5)
        assert model.hidden_dim == 32
        assert model.num_layers == 1
        assert model.rnn.input_size == 10
        assert model.fc.out_features == 5
    
    def test_rnn_forward_pass(self):
        """Test RNN forward pass"""
        model = RNNModel(input_dim=10, hidden_dim=32, output_dim=5)
        batch_size = 2
        seq_length = 12
        input_dim = 10
        
        # Create dummy input
        x = torch.randn(batch_size, seq_length, input_dim)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestLSTMModel:
    """Test LSTM model implementation"""
    
    def test_lstm_initialization(self):
        """Test LSTM model initialization"""
        model = LSTMModel(input_dim=10, hidden_dim=32, output_dim=5)
        assert model.hidden_dim == 32
        assert model.num_layers == 1
        assert model.lstm.input_size == 10
        assert model.fc.out_features == 5
    
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass"""
        model = LSTMModel(input_dim=10, hidden_dim=32, output_dim=5)
        batch_size = 2
        seq_length = 12
        input_dim = 10
        
        # Create dummy input
        x = torch.randn(batch_size, seq_length, input_dim)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestGRUModel:
    """Test GRU model implementation"""
    
    def test_gru_initialization(self):
        """Test GRU model initialization"""
        model = GRUModel(input_dim=10, hidden_dim=32, output_dim=5)
        assert model.hidden_dim == 32
        assert model.num_layers == 1
        assert model.gru.input_size == 10
        assert model.fc.out_features == 5
    
    def test_gru_forward_pass(self):
        """Test GRU forward pass"""
        model = GRUModel(input_dim=10, hidden_dim=32, output_dim=5)
        batch_size = 2
        seq_length = 12
        input_dim = 10
        
        # Create dummy input
        x = torch.randn(batch_size, seq_length, input_dim)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestTransformerModel:
    """Test Transformer model implementation"""
    
    def test_transformer_initialization(self):
        """Test Transformer model initialization"""
        model = TransformerModel(
            input_dim=10, 
            d_model=32, 
            nhead=4, 
            num_layers=1, 
            output_dim=5, 
            seq_length=12
        )
        assert model.d_model == 32
        assert model.nhead == 4
        assert model.input_proj.in_features == 10
        assert model.output_layer.out_features == 5
    
    def test_transformer_forward_pass(self):
        """Test Transformer forward pass"""
        model = TransformerModel(
            input_dim=10, 
            d_model=32, 
            nhead=4, 
            num_layers=1, 
            output_dim=5, 
            seq_length=12
        )
        batch_size = 2
        seq_length = 12
        input_dim = 10
        
        # Create dummy input
        x = torch.randn(batch_size, seq_length, input_dim)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestModelConsistency:
    """Test model consistency and edge cases"""
    
    def test_all_models_same_input_output(self):
        """Test that all models handle same input/output dimensions"""
        input_dim = 10
        output_dim = 5
        batch_size = 2
        seq_length = 12
        
        # Create models
        rnn = RNNModel(input_dim, 32, output_dim)
        lstm = LSTMModel(input_dim, 32, output_dim)
        gru = GRUModel(input_dim, 32, output_dim)
        transformer = TransformerModel(input_dim, 32, 4, 1, output_dim, seq_length)
        
        # Create dummy input
        x = torch.randn(batch_size, seq_length, input_dim)
        
        # Test all models
        models = [rnn, lstm, gru, transformer]
        for model in models:
            output = model(x)
            assert output.shape == (batch_size, output_dim)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_model_device_consistency(self):
        """Test model device consistency"""
        model = RNNModel(10, 32, 5)
        
        # Test CPU
        x_cpu = torch.randn(2, 12, 10)
        output_cpu = model(x_cpu)
        assert output_cpu.device.type == 'cpu'
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            x_gpu = x_cpu.cuda()
            output_gpu = model_gpu(x_gpu)
            assert output_gpu.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__])
