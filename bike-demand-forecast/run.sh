
#!/bin/bash

set -e

echo "Bike Demand Forecasting Pipeline"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$PROJECT_DIR"

# Python

if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: Python 3 was not found."
    exit 1
fi

echo "Using Python: $PYTHON_BIN"

# Testing

echo ""
echo "-------------------- TESTING --------------------"

"$PYTHON_BIN" -m pytest -q

# Data preparation

echo ""
echo "-------------------- DATA PREPARATION --------------------"

"$PYTHON_BIN" -m src.data

# Model training

echo ""
echo "-------------------- MODEL TRAINING --------------------"

"$PYTHON_BIN" -m src.train

# Model evaluation

echo ""
echo "-------------------- MODEL EVALUATION --------------------"

"$PYTHON_BIN" -m src.evaluate

# Model selection

echo ""
echo "-------------------- MODEL SELECTION --------------------"

"$PYTHON_BIN" -m src.model_selection

# Final result

echo ""
echo "-------------------- PIPELINE COMPLETE --------------------"

echo ""
echo "Selected model:"

"$PYTHON_BIN" -c "
import json

with open('models/metadata.json') as f:
    metadata = json.load(f)

print(metadata['selected_model'])
"

echo ""
echo "To start Streamlit:"
echo ""
echo "$PYTHON_BIN -m streamlit run app.py"

