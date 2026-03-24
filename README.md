# tradIA — AI Trading Intelligence System

An AI-powered trading intelligence system that learns ICT/SMC (Smart Money Concepts) patterns from BTC/USDT historical price data and generates high-precision **BUY/SELL** signals.

## Project Structure

```
tradIA/
├── data/                       # Dataset directory
│   └── FINAL (1).csv          # BTC/USDT 15m OHLC data (2022-2025)
├── src/                        # Source modules
│   ├── config.py              # Central configuration
│   ├── data_loader.py         # Data loading & validation
│   ├── feature_engineering.py # ICT/SMC feature extraction
│   ├── label_generator.py     # Triple-barrier labeling
│   ├── model_lightgbm.py      # LightGBM pipeline
│   ├── model_lstm.py          # LSTM deep learning alternative
│   ├── evaluation.py          # Metrics & signal quality
│   └── interpretability.py    # SHAP feature importance
├── models/                     # Saved model artifacts
├── results/                    # Evaluation outputs & plots
├── run_pipeline.py            # Main entry point
└── requirements.txt           # Python dependencies
```

## Quick Start

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run the full pipeline
python run_pipeline.py

# Skip Optuna optimization (faster, uses defaults)
python run_pipeline.py --skip-optuna

# Skip LSTM training
python run_pipeline.py --skip-lstm

# Custom Optuna trials
python run_pipeline.py --optuna-trials 20
```

## Features

The system engineers 70+ features from raw OHLC data based on ICT/SMC concepts:

- **Swing Structure**: Swing highs/lows, distance and time since last swing
- **Market Structure**: Break of Structure (BOS), Market Structure Shifts (MSS)
- **Fair Value Gaps**: Bullish/bearish FVG detection, active count, distance
- **Liquidity Zones**: Equal highs/lows clusters, liquidity sweep detection
- **Displacement**: ATR-based displacement detection, impulsive moves
- **CISD**: Change in State of Delivery detection
- **Order Flow**: Consecutive candle analysis, buying/selling pressure
- **Multi-Timeframe**: 1H and 4H structure alignment
- **Context**: Session-based features, daily range position

## Models

- **LightGBM** (Primary): Gradient-boosted trees with Optuna optimization
- **LSTM** (Alternative): 2-layer LSTM for sequential pattern recognition

Both use walk-forward validation to prevent data leakage.

## License

This project is part of a Final Year Project (FYP) in Computer Science.
