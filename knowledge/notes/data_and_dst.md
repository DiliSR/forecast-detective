# The dataset and how it was prepared

## What the data is

The UCI ElectricityLoadDiagrams20112014 dataset (UCI id 321) holds electricity consumption of 370 clients in Portugal,
recorded every 15 minutes in kW from 2011 to the end of 2014. Many clients started later than 2011 (their earlier readings are zero).
This project forecasts the **total hourly consumption in kWh of the 319 clients that have data throughout 2012-2014**.
Forecasting the sum of many clients is much smoother than forecasting one client, which is why errors of about 3% are possible.

## Why the readings are divided by 4 to get kWh

A reading is the average power in kW over 15 minutes. Energy over 15 minutes is power times 0.25 hours.
Summing the four readings of an hour gives kW multiplied by the number of readings; dividing by 4 converts that sum to kWh.
The timestamps are end-labelled (the 01:00 reading covers 00:45 to 01:00), so the series is shifted by 15 minutes
before grouping into hours, otherwise the 01:00 reading would be counted in the wrong hour.

## Daylight saving time (clock change)

Portugal changes the clocks on the last Sunday of March (clocks go forward) and the last Sunday of October (clocks go back).
The raw file does not handle this consistently. The four 15-minute rows labelled 01:00, 01:15, 01:30 and 01:45 on those days are wrong:

- March: that hour does not exist, so the rows hold only about 4,000 instead of about 120,000. They are replaced by linear interpolation between 00:45 and 02:00.
- October: that hour happens twice, and both passes are summed into one row, so the rows hold about twice the normal value. They are divided by 2.

The repair is applied to the 15-minute rows before hourly aggregation. In the test period the October 2014 clock change (26 October) is included,
and the hours around it are forecast normally.

## Train, validation and test periods

- Training: up to 31 December 2013 23:00 (the first four weeks of 2012 are dropped because the 4-week average feature needs history).
- Validation: January to June 2014, used only for early stopping.
- Test: July to December 2014, never used for fitting or tuning. All accuracy numbers in this project are on the test period.

## Day-ahead forecasting

Every feature that uses load is at least 24 hours old, so a forecast for hour t uses only information available a day before.
The models predict each hour independently from actual past loads; they do not feed their own forecasts back in,
so errors do not accumulate over the day. Error depends on the time of day (RMSE at night about 3,200 kWh for LightGBM and 4,300 for the LSTM; afternoon about 13,700 and 13,200; evening about 12,100 and 12,600), not on how far ahead the hour is.
A realistic caveat: the Iberian day-ahead market closes around midday, so a real deployment may only know loads up to about 36 hours earlier.
