# SKUAST Tree Analysis System

## Modules

- `chemical_analysis.py`: Main script for chemical analysis (use `/analyze/chemical` endpoint)
- `yield_analysis.py`: Main script for yield analysis (use `/analyze/yield` endpoint)

## Usage

- For chemical analysis, use the `/analyze/chemical` endpoint.
- For yield analysis, use the `/analyze/yield` endpoint.

See the code for details on input parameters and output.

## Features
- Chemical compound analysis
- Tree species comparison
- Data visualization
- Statistical analysis of chemical properties

## Installation
1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
- `data/`: Directory for storing tree chemical data
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Data Format
The project expects chemical data in CSV format with the following columns:
- Tree Species
- Chemical Compound
- Concentration
- Measurement Date
- Location 