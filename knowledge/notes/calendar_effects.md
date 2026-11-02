# Calendar effects on Portuguese electricity load

Measured on the UCI ElectricityLoadDiagrams20112014 data (total load of the 319 full-coverage clients).
Each percentage compares the day with the **same weekday one and two weeks earlier** (ordinary days),
for 2012, 2013 and 2014 where an earlier reference exists. Ranges are the lowest and highest of those years.

## Christmas Eve (24 December) is not an official holiday but the evening load drops sharply

24 December is **not** in the Portuguese public-holiday calendar (python-holidays `Portugal`), so neither
model's `is_holiday` or `is_major_holiday` flag is set for it. The load nevertheless changes through the day:

- 12:00: +4% to +9% (slightly above a normal day)
- 18:00: -9% to -16%
- 20:00: -29% to -43%
- whole day: only -4% to -7%, because the morning and midday are normal

So the damage is concentrated in the evening. The pattern repeated in 2012, 2013 and 2014.
Interpretation (not tested with this data): consistent with people stopping work early and spending the evening at home.

Consequence for the forecasts: both models forecast Christmas Eve evenings as ordinary working days.
On 24 Dec 2014 at 18:00 the actual load was 211,763 kWh while LightGBM forecast 242,680 kWh (+14.6%) and the LSTM 243,750 kWh (+15.1%).
At 20:00 the over-forecast was about +47% (LightGBM) and +50% (LSTM).

## Christmas Day (25 December) is a major holiday and load falls by about 40%

25 December is a public holiday and the load for the whole day is far below normal:

- whole day: -34% to -43% (2012: -34%, 2013: -37%, 2014: -43%)
- 18:00: -36% to -47%

This is by far the largest calendar effect in the data. It is why the models have a dedicated `is_major_holiday`
feature (25 Dec and 1 Jan), separate from the general `is_holiday` flag. Ordinary public holidays do not have this effect.
Even with that feature the models are less accurate on major holidays than on any other day type:
RMSE on 25 December 2014 (the only major holiday in the test set) is 32,004 kWh for LightGBM and 17,490 kWh for the LSTM, against about 10,400 and 10,800 kWh on ordinary days.

## New Year's Eve (31 December) drops in the late evening only

31 December is not a public holiday. Load stays normal until the evening and then falls:

- 12:00 and 18:00: within about 6% of a normal day (-6% to +1%)
- 20:00: -23% to -29%
- whole day: -2% to -7%

On 31 December 2014 both models over-forecast the 20:00 and 21:00 hours (LightGBM +31% and +38%, LSTM +55% and +53%)
because no flag marks the day as special. The error comes from missing calendar information, not from the load history.

## New Year's Day (1 January) behaves like a major holiday

1 January is a public holiday and the whole-day load is -24% (2013) and -21% (2014) relative to the same weekday
one and two weeks earlier. This is a smaller drop than Christmas Day. 1 January 2012 has no earlier reference in the data.

## Ordinary Portuguese public holidays barely change the load

Weekday public holidays other than 25 Dec and 1 Jan (e.g. Good Friday, 25 April, 1 May, 10 June, 15 August, 8 December)
change the whole-day load by between about -11% and +7% compared with the same weekday in the previous two weeks,
with no consistent direction across the 18 cases in 2012-2014. The load of these days looks like an ordinary working day,
The test set contains one bridge day: RMSE 6,334 kWh for LightGBM and 11,696 kWh for the LSTM, against about 10,400 and 10,800 kWh on ordinary days. One day is too few to conclude anything.

## Bridge days (pontes)

A Monday before a Tuesday holiday or a Friday after a Thursday holiday is often taken off. The models have an
`is_bridge_day` feature for this. It fires on only 10 days in 2012-2014 and its effect is small compared with Christmas.
In 2012 it happened to mark 24 and 31 December (Mondays before Tuesday holidays). In 2013 and 2014 those days fell on a
Tuesday and a Wednesday and were not flagged. The flag is therefore set on Christmas Eve in only one of the three years,
so it cannot teach the model that 24 December is special.

## Weather is not used

Neither model has any weather input (temperature, daylight). Load changes caused by unusual weather appear as forecast
error that no feature can explain. Weather covariates are a possible future extension.
