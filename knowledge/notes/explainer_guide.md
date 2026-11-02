# How to read each explanation method in this project

Attributions are in kWh: how much an input moved this forecast away from the method's **reference point**.
The reference points differ between methods, which is why methods can legitimately report different numbers.

## SHAP (LightGBM)

SHAP uses exact TreeSHAP. The reference is the model's **average forecast over all training hours** (day and night, all seasons),
reported as the expected value. The attributions plus the expected value add up exactly to this forecast.
Because the reference mixes night and day, a feature such as `hour` often receives a large attribution simply because
the hour being forecast is a busy one, not because anything unusual happens.

## LIME (LightGBM)

LIME fits a weighted linear model to many perturbed copies of the input around this one forecast.
It has no reference point: its weights are slopes of that local line, not a split of the forecast.
It is random, so we run several seeds and report the spread and the rank range of each feature.
The surrogate R-squared says how much of the model's local behaviour the line captures.
A low R-squared (below about 0.3) means the line does not describe the model well here and the weights are weak evidence.
Tree models change abruptly at splits such as holiday flags, which is hard for a straight line to follow, so LIME fits poorly near holidays.

## Integrated Gradients (LSTM)

Integrated Gradients accumulates the gradient of the forecast along a straight path from a **baseline input** to the actual input.
The attributions sum to the forecast minus the baseline forecast (completeness). The convergence delta is the numerical error of that sum.
Here the baseline is a typical non-holiday weekday week ending at the same hour of day and in the same season (target month plus or minus one month),
with all holiday and weekend flags switched off. The question answered is therefore "why is this forecast different from a normal weekday at this hour and time of year?".
A zero baseline was rejected because it credits hour-of-day features with explaining that the target is an hour at all.

## TimeSHAP (LSTM)

TimeSHAP applies Shapley values to a sequence model by switching parts of the input between the actual value and the baseline.
It reports three levels: events (which past hours), features (which input features) and cells (a specific hour and feature together).
To keep the estimation tractable it first prunes the oldest hours whose joint effect is below a tolerance (500 kWh here) into one
group called older hours. It uses the same baseline as Integrated Gradients, so the two LSTM explainers are directly comparable.
Because the target-hour flags are repeated on every step, an event total includes part of the effect of the target-day flags;
the cell level separates them.

## Why the explainers can disagree

- Different references: SHAP is relative to the average over all hours, IG and TimeSHAP to a typical weekday at the same hour, LIME to nothing.
- Different quantities: LIME weights are slopes, SHAP values are shares of the forecast.
- Randomness: LIME and TimeSHAP sample, so repeated runs differ slightly.
- Correlated inputs: lag features such as load 24 hours ago and the 24-hour mean carry similar information, so credit can be split between them in different ways.

## How the disagreement flags are computed

The project compares explainers by **rank** (Spearman correlation of the ranking of absolute attributions, restricted to features that matter,
meaning at least 2% of the forecast) and by **direction** (a feature that one method says pushes the forecast up and another says pushes it down).
It also flags when the two models forecast more than 3% apart, when both models are more than 10% away from the actual in the same direction
(information neither model has), when LIME's R-squared is below 0.3, and when the LIME ranking changes between random seeds.
