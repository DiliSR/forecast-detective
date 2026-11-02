# Model cards

## LightGBM day-ahead model (lgbm_total)

A gradient-boosted decision-tree regressor that forecasts the total hourly load (kWh) one day ahead.

- Inputs (21 features): load lags (24, 25, 48, 72, 168, 336 hours), rolling mean and standard deviation over 24 hours,
  rolling mean over 7 days, the average of the same hour over the last four weeks, flags telling whether the lagged day
  (24 h and 168 h ago) was a holiday or major holiday, and calendar features for the target hour
  (hour, day of week, month, weekend, holiday, major holiday, bridge day).
- All load features are at least 24 hours old (day-ahead). No weather. No client-level information.
- Training: data up to 31 Dec 2013. Early stopping on January-June 2014 chose 284 trees. Learning rate 0.03, 63 leaves, minimum 20 samples per leaf.
- Test accuracy (July-December 2014): RMSE 10,296 kWh (the headline error measure), MAE 6,731 kWh, MAPE 3.03%.
  Naive baselines: same hour yesterday 3.78%, same hour last week 5.44%, four-week same-hour mean 6.15%.
- Known weaknesses: major holidays (RMSE 32,004 kWh on 25 December, forecasts too high by about 27% on average) and holiday-like days with no flag
  (Christmas Eve and New Year's Eve evenings). It leans heavily on recent load, so it copies yesterday's and last week's pattern.
- Explained with: SHAP (exact TreeSHAP) and LIME.

## LSTM day-ahead model (lstm_total)

A small recurrent neural network (one LSTM layer with 64 hidden units followed by a linear layer) forecasting the same target.

- Input window: a sequence of the 168 hours (one week) ending 24 hours before the target hour. Each step of the window carries the standardised load,
  hour of day as sine and cosine, and holiday and major-holiday flags of that past hour. Three flags about the target hour
  (weekend, holiday, major holiday) are repeated on every step.
- Training: same period and split as LightGBM, Adam optimiser, learning rate 0.001, early stopping with patience 5.
- Test accuracy (July-December 2014): RMSE 10,389 kWh, MAE 7,078 kWh, MAPE 3.28%. Slightly worse than LightGBM overall.
- Known weaknesses: also fails on days without a holiday flag. It is better than LightGBM on major holidays (RMSE 17,490 vs 32,004 kWh on 25 December, a single day)
  but worse on ordinary weekdays (RMSE 10,835 vs 10,372 kWh) and at night (4,307 vs 3,169 kWh).
- It cannot see anything older than one week, whereas LightGBM also uses a 336-hour lag and a four-week average.
- Explained with: Integrated Gradients (Captum) and TimeSHAP.

## Comparing the two

On the test period the models score similarly (3.03% vs 3.28%), and the largest errors of both occur on the same hours:
the evenings of 24 December and 31 December and the major holidays. When both models miss in the same direction,
the cause is information that neither model has (for example a missing holiday flag), not random model noise.
