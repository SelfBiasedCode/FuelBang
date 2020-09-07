# FuelBang

## Summary
FuelBang is a python script to increase engine braking and add deceleration pops for Triumph motorcycles. It relies on the CSV import and export function of the Android App TuneECU. 

## Technical Details
During engine braking, a negative (braking) torque is created by the engine itself, mainly due to air compression and friction losses. This is offset by any fuel burned in this state, which will create a positive (driving) torque. Therefore, the amount of engine braking can be controlled either by using the throttle or by controlling the amount of fuel injected at a given throttle position.
In general, the throttle will be closed during engine braking, creating a high intake vacuum. Also the engine sspeeed will be above idle. This can be used to change the fueling for just the engine braking areas of the mapping: Triumph Keihin ECUs use "F"-Maps for RPM & Throttle position above a certain throttle percentage. Below that, "L"-tables for RPM & intake pressure are used. This script therefore modifies only the "L"-Maps and only in an area defined by typical engine braking pressure and a minimum RPM that is significantly above idle RPM.

## Usage
1. Edit values in the script and run.
2. OutputView.xlsx will contain a comparison between baseline, previous and current L1 table.
3. Output L1 and L2 files will be in the *Data* folder.

## Known Issues
No error checking is performed.

## TODO
* encapsulate class
* add config object
* add docopt