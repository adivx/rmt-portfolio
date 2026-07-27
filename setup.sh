#!/bin/bash
# RMT Portfolio Optimization — One-click setup

echo "🔧 Setting up RMT Portfolio Optimization project..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.9+"
    exit 1
fi

echo "✅ Python $(python3 --version | cut -d' ' -f2) detected"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Generate notebook
echo "📓 Generating Jupyter notebook..."
python3 create_notebook.py

# Create data directory
mkdir -p data output figures

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "   1. cd $(pwd)"
echo "   2. jupyter notebook notebooks/rmt_portfolio_optimization.ipynb"
echo ""
echo "   Or run directly:"
echo "   python3 src/backtester.py"
