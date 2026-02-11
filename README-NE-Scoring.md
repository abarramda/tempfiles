# NE Model – Cleveland Scoring

This workflow fits an OLS model on the logit of Market penetration (fraction) using the NE training dataset, then scores the Cleveland CSV and produces an Excel workbook with:
- Readme
- Training summary
- Model spec
- Coefficients (robust SE)
- Diagnostics (CV, residuals)
- Scoring – Cleveland (predictions and features)
- Charts

Data sources:
- Training: Demand MVR & Nonlinear_lite NE.csv (provided by abarramda)
- Scoring: Regression test - Cleveland.csv (provided by abarramda)

Modeling choices:
- Target: Market penetration fraction p = penetration%/100
- Transform: logit(p); clamp p to [1e-6, 1 - 1e-6]
- Features (intersection rule): 
  - Distance: Beeline distance (mi) with fallback to Road distance (mi) if missing per-row
  - eVTOL time (min), Road time (min) → time_adv = Road time − eVTOL time
  - eVTOL cost (USD), eVTOL cost per mile (USD/mi)
  - Income band A–E trips share (stored as fractions)
  - Commute/Personal/Business trip shares are excluded in scoring (fully empty); included in training only if NON-empty for a row
- Regularization: none (OLS)
- SE: heteroskedasticity-robust (HC3)
- CV: 5-fold (R², RMSE, MAE on fraction scale)

Output:
- NE Model – Cleveland Scoring.xlsx (saved at repo root by default)

Run:
- Trigger the workflow "NE model scoring" manually (Actions → NE model scoring → Run workflow)
