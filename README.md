# Portfolio Rebalancer

Tool for investment portfolio rebalancing, long-term wealth planning and housing-purchase savings forecasting.

Production deployment:  
https://ibanez-capital.streamlit.app/

## Overview

Portfolio Rebalancer is a full financial-planning environment built for retail investors, personal-finance optimisation and fintech experimentation.

It provides:

· Portfolio construction and rebalancing  
· Automatic monthly contribution optimisation  
· Overweight detection and suggested sales  
· Long-term wealth projection (constant or growing contributions)  
· Mortgage and down-payment planning  
· Custom-asset management and an enriched asset universe  
· Professional data visualisation and reporting

## Key Features

### 1. Portfolio Rebalancing

· Add assets with name, ISIN, type, value and target weights  
· Integrated search engine using ISIN, name or normalised keywords  
· Generate optimal monthly contribution distribution  
· Compare target vs current allocations  
· Identify overweight positions and compute suggested sales  
· Save, load and delete named portfolios  
· Import custom assets into the universe

### 2. Long-Term Wealth Goal Planning

· Set a future wealth target  
· Choose constant or growing monthly contributions  
· Model long-term expected returns  
· Optionally account for capital-gains tax using progressive brackets  
· Estimate required gross and net salary based on desired saving rate  
· Visualise projected portfolio growth

### 3. Housing-Purchase Savings Plan

· Compute required down payment and transaction costs  
· Estimate years of required saving  
· Simulate constant or increasing monthly contributions  
· Optionally apply capital-gains taxation  
· Generate approximate mortgage payments

## Technology Stack

Python, Streamlit, Pandas, NumPy, Matplotlib, Altair, OpenPyXL.

## Repository Structure

· `app.py`: Main Streamlit application

· `asset_universe.csv`: Unified asset universe used for search and autocompletion

· `asset_universe_builder.py`: Script to build and normalise the asset universe

· `carteras.json`: Storage for saved user portfolios

· `planes.json`: Storage for saved long-term financial and housing plans

· `activos_custom.json`: Storage for user-defined custom assets

· `Instrument_Universe_DE_en.pdf`: Source reference for the constructed asset universe

· `requirements.txt`: Python dependencies required to run the application


## Installation

Clone repository:

```
git clone https://github.com/MarcosIbanezFandos/Portfolio-Rebalancer.git
cd Portfolio-Rebalancer
```

Install dependencies:

```
pip install -r requirements.txt
```


Run application:

```
streamlit run app.py
```


## Rebalancing Logic (Simplified)

The system applies:

· Weight calculations based on market values  
· Ideal contribution distribution  
· No negative contributions (no implicit sales)  
· Redistribution of remaining capital  
· Recalculation of post-contribution weights  
· Overweight detection beyond deviation threshold  
· Minimal-sales calculation

Example call:

```
contrib_plan = compute_contribution_plan(
portfolio=port,
monthly_contribution=monthly_contribution,
rebalance_threshold=rebalance_threshold
)
```

## Asset Universe & Search Engine

The app includes an extended asset universe with:

· ISIN  
· Name  
· Type  
· Region and Country (when available)  
· ETF provider and subtype  
· Currency  
· Normalised search keys

Special logic allows matching strings like S&P 500 even when searched without "&" (e.g. "sp").

Users may extend the universe through:

```
activos_custom.json
```


## Saving & Loading Portfolios

Saved portfolios are stored in:

```
carteras.json
```

Each entry contains asset name, ISIN, type, value and target weight.

This enables managing multiple independent portfolios.

## Deployment

The production system runs on Streamlit Cloud.

To self-host:

· Install dependencies  
· Upload universe + JSON files  
· Run:

```
streamlit run app.py
```
