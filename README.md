# 🚗 Traffic Flow Prediction using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning project for predicting traffic flow patterns using various deep learning and traditional ML models. This project implements and compares multiple approaches including RNN, LSTM, GRU, Transformer, SVM, and Logistic Regression on real-world traffic datasets.

## 📊 Project Overview

This project aims to predict future traffic flow patterns using historical sensor data from major metropolitan areas. It provides a complete pipeline from data preprocessing to model evaluation, with interactive visualizations and performance comparisons.

### 🎯 Key Features

- **Multiple Model Architectures**: RNN, LSTM, GRU, Transformer, SVM, Logistic Regression
- **Real-world Datasets**: METR-LA (Los Angeles) and PEMS-BAY (San Francisco Bay Area)
- **Interactive Dashboard**: Streamlit-based visualization and analysis tool
- **Comprehensive Evaluation**: RMSE, MAE, R², Accuracy, AUC metrics
- **Time Series Prediction**: 12-step input sequences to predict 3 future time steps
- **Professional Documentation**: Complete setup and usage instructions

## 🏗️ Project Structure

```
traffic-flow-prediction/
├── 📁 data/                          # Dataset files
│   ├── METR-LA.csv                   # Los Angeles traffic data (CSV)
│   ├── METR-LA.npz                   # Los Angeles traffic data (NumPy)
│   ├── PEMS-BAY.csv                  # San Francisco Bay Area data (CSV)
│   └── PEMS-BAY.npz                  # San Francisco Bay Area data (NumPy)
├── 📁 src/                           # Source code
│   ├── main.py                       # Main training and evaluation script
│   ├── main1.py                      # Simplified version of main script
│   ├── inference.py                  # Model inference and prediction script
│   ├── backup.py                     # Data conversion utility
│   └── datasetformatter.py           # Dataset preprocessing utility
├── 📁 scripts/                       # Utility scripts
│   ├── streamlit_dashboard.py        # Interactive dashboard application
│   └── run_dashboard.py              # Dashboard launcher script
├── 📁 models/                        # Trained model files (generated)
├── 📁 figures/                       # Generated visualizations
│   ├── GRU_traffic_flow_predictions.png
│   └── Logistic Regression_logistic_regression_visualization.png
├── 📁 docs/                          # Documentation and results
│   ├── results.json                  # Model performance results
│   └── execution.txt                 # Command execution log
├── 📁 notebooks/                     # Jupyter notebooks (for future use)
├── 📁 docs/                          # Additional documentation
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/traffic-flow-prediction.git
   cd traffic-flow-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main training script:**
   ```bash
   python src/main.py --dataset data/METR-LA.npz --type metr_la
   ```

4. **Launch the interactive dashboard:**
   ```bash
   python scripts/run_dashboard.py
   ```

## 📊 Datasets

### METR-LA Dataset
- **Location**: Los Angeles Metropolitan Area
- **Sensors**: 207 traffic sensors
- **Time Period**: March 1, 2012 - June 30, 2012 (4 months)
- **Sampling Rate**: 5-minute intervals
- **Data Points**: ~34,272 timesteps

### PEMS-BAY Dataset
- **Location**: San Francisco Bay Area
- **Sensors**: 325 traffic sensors
- **Time Period**: January 1, 2017 - May 31, 2017 (6 months)
- **Sampling Rate**: 5-minute intervals
- **Data Points**: ~52,128 timesteps

## 🤖 Models Implemented

### Deep Learning Models

| Model | Architecture | Hidden Units | Key Features |
|-------|-------------|--------------|--------------|
| **RNN** | Recurrent Neural Network | 32 | Simple sequential processing |
| **LSTM** | Long Short-Term Memory | 32 | Memory cells, long-term dependencies |
| **GRU** | Gated Recurrent Unit | 32 | Simplified LSTM, faster training |
| **Transformer** | Attention-based | 32 (d_model), 4 heads | Parallel processing, attention mechanism |

### Traditional ML Models

| Model | Type | Configuration | Purpose |
|-------|------|---------------|---------|
| **SVM** | Support Vector Machine | Linear kernel, Multi-output | Regression prediction |
| **Logistic Regression** | Binary Classification | Threshold-based | Traffic flow classification |

## 📈 Performance Results

### Model Performance Comparison

| Model | RMSE | MAE | R² Score | Loss |
|-------|------|-----|----------|------|
| **LSTM** | 10.46 | 6.19 | 0.639 | 0.370 |
| **GRU** | 10.34 | 6.16 | 0.637 | 0.373 |
| **RNN** | 10.69 | 6.52 | 0.609 | 0.402 |
| **Transformer** | 11.68 | 6.96 | 0.539 | 0.473 |
| **Logistic Regression** | - | - | - | Accuracy: 74%, AUC: 0.817 |

### Key Insights
- **Best Model**: LSTM with R² score of 0.639
- **Close Second**: GRU with R² score of 0.637
- **Deep Learning Advantage**: All neural networks outperformed traditional ML
- **LSTM Superiority**: Best at capturing long-term traffic patterns

## 🎛️ Usage

### Training Models

```bash
# Train on METR-LA dataset
python src/main.py --dataset data/METR-LA.npz --type metr_la

# Train on PEMS-BAY dataset
python src/main.py --dataset data/PEMS-BAY.npz --type pems_bay

# Load saved models instead of training
python src/main.py --dataset data/METR-LA.npz --type metr_la --load
```

### Making Predictions

```bash
# Generate future predictions
python src/inference.py --model models/lstm_model.pth --dataset data/METR-LA.npz --type metr_la --nodes 5 --horizon 12
```

### Interactive Dashboard

```bash
# Launch the Streamlit dashboard
python scripts/run_dashboard.py

# Or directly with Streamlit
streamlit run scripts/streamlit_dashboard.py
```

## 📊 Dashboard Features

The interactive Streamlit dashboard provides:

- **📈 Performance Visualization**: Interactive charts comparing all models
- **📊 Data Exploration**: Traffic pattern analysis and statistics
- **🏗️ Architecture Details**: Model structure and configuration
- **🔮 Prediction Demo**: Live prediction demonstrations
- **📱 Responsive Design**: Works on desktop, tablet, and mobile

### Dashboard Sections

1. **Overview** - Project summary and key statistics
2. **Dataset Info** - Detailed dataset information
3. **Model Performance** - Interactive performance comparison
4. **Architecture** - Model architecture explanations
5. **Visualizations** - Data and prediction visualizations
6. **Predictions** - Sample prediction demonstrations
7. **Conclusions** - Key insights and project summary

## 🔧 Configuration

### Model Parameters

```python
# Sequence configuration
seq_length = 12      # Input sequence length (1 hour)
pred_length = 3      # Prediction length (15 minutes)

# Training parameters
batch_size = 128
learning_rate = 0.001
num_epochs = 10
patience = 2         # Early stopping patience

# Model architecture
hidden_dim = 32      # Hidden units for RNN/LSTM/GRU
d_model = 32         # Transformer model dimension
nhead = 4            # Transformer attention heads
```

### Data Preprocessing

- **Normalization**: StandardScaler for input and output data
- **Sequence Creation**: Sliding window approach
- **Train/Val/Test Split**: 70%/15%/15% split
- **Memory Optimization**: Limited to 10,000 samples for efficiency

## 🧪 Experimentation

### Adding New Models

1. Create model class in `src/main.py`
2. Add training logic in the main function
3. Include evaluation metrics
4. Update dashboard visualization

### Customizing Datasets

1. Add dataset loading function
2. Update preprocessing pipeline
3. Modify model input dimensions
4. Test with new data format

## 📊 Results and Visualizations

The project generates several types of visualizations:

- **Training History**: Loss curves for each model
- **Prediction Plots**: Actual vs predicted traffic flow
- **Performance Comparison**: Bar charts comparing metrics
- **Traffic Patterns**: Heatmaps and time series plots
- **Model Architecture**: Visual representations of model structures

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/traffic-flow-prediction.git
cd traffic-flow-prediction
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Run linting
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **METR-LA Dataset**: Li et al. "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting"
- **PEMS-BAY Dataset**: California Department of Transportation
- **PyTorch**: Deep learning framework
- **Streamlit**: Interactive dashboard framework
- **scikit-learn**: Machine learning utilities

## 📞 Contact

- **Author**: Shuban Borkar
- **Email**: shubanborkar@outlook.com
- **GitHub**: https://github.com/shubanborkar
- **LinkedIn**: https://www.linkedin.com/in/shuban-borkar/

## 🔗 Related Projects

- [Traffic Prediction with Graph Neural Networks](https://github.com/example/gnn-traffic)
- [Urban Mobility Analysis](https://github.com/example/urban-mobility)
- [Smart City Traffic Management](https://github.com/example/smart-traffic)

## 📚 References

1. Li, Y., et al. "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting." ICLR 2018.
2. Yu, B., et al. "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting." IJCAI 2018.
3. Vaswani, A., et al. "Attention Is All You Need." NIPS 2017.

---

**⭐ If you found this project helpful, please give it a star!**

*Last updated: October 2025*
