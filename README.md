# NIFTY Risk Terminal

A browser-based risk analytics terminal for NIFTY (or any instrument). Upload an OHLC price CSV and it computes everything client-side, with no backend and no API keys.

## Features

| Tab | What it shows |
|---|---|
| **Price & Returns** | Price series, log returns, return distribution histogram, skewness and kurtosis |
| **VaR / CVaR** | Historical Value-at-Risk and Conditional VaR (Expected Shortfall) |
| **Volatility** | Rolling volatility, EWMA (λ = 0.94) and GARCH(1,1) estimates, annualised |
| **HMM Regimes** | 3-state Gaussian Hidden Markov Model fitted with Baum–Welch, labelling Bull / Sideways / Bear regimes |
| **Options Greeks** | Black–Scholes pricer with Delta, Gamma, Vega, Theta and Rho for calls and puts |

## Getting started

```bash
npm install
npm run dev
```

Then open the local URL Vite prints and upload a CSV with a `date` column and a `close` column. Column names are matched loosely, and ISO-8601 timestamps are handled.

## Tech stack

React 19, Vite and Recharts. All the models (Baum–Welch HMM, GARCH, EWMA, VaR/CVaR, Black–Scholes) are implemented from scratch in `src/App.jsx`.

## Disclaimer

For educational and research use only. Not investment advice.
