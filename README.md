# FuelBang

## Disclaimer
This script is a tool to directly modify fuel tables. Reflashing your bike's ECU can lead to accidents, engine damage, increased emissions reduced performance, reduced gas mileage and other unintended consequences. Doing so may be illegal depending on where you live. Use at your your own risk and liability.

## Summary
FuelBang is a python script to increase engine braking and add deceleration pops for Triumph motorcycles. It relies on the CSV import and export function of the Android App TuneECU. The script works by defining an engine braking region by RPM and intake pressure limits and replacing the fuel values in that region with zeros, then smoothes out the region's edges with a selectable mathematical algorithm. 

## Technical Details
During engine braking, a negative (braking) torque is created by the engine itself, mainly due to air compression and friction losses. This is offset by any fuel burned in this state, which will create a positive (driving) torque. Therefore, the amount of engine braking can be controlled either by using the throttle or by controlling the amount of fuel injected at a given throttle position.
In general, the throttle will be closed during engine braking, creating a high intake vacuum. Also the engine speed will be above idle. This can be used to change the fueling for just the engine braking areas of the mapping: Triumph Keihin ECUs use "F"-Maps for RPM & Throttle position above a certain throttle percentage. Below that, "L"-tables for RPM & intake pressure are used. This script therefore modifies only the "L"-Maps and only in an area defined by typical engine braking pressure and a minimum RPM that is significantly above idle RPM.

## Usage
1. Get baseline L1 and L2 tables by exporting them from a stock map with TuneECU.
2. Determine the engine braking region you want to modify and choose smoothing parameters. (see below for tips)
3. Run the script. It will create modified L1 and L2 tables plus comparisons to the baseline.
4. Use a spreadsheet application of your choice to view the baseline, processed and difference tables.
5. For further modifications, choose the stock tables as baseline and your previous modifications as previous.

## Parametrization
Building fuel tables completely based on mathematics is pretty much impossible, modifying them even more so. The reasons for this are:
- Every engine is different. Take a look at the stock fuel tables, they will be anything but linear, symmetric or otherwise trivial to explain. A lot of variables come into account for optimal performance, emissions and longevity of the components.
- Every target is different. Most people who modify their fuel tables have done something to their engine, from changing the exhaust to a complete overhaul of the internals. It is impossible to make generic assumptions on how changing a component will affect fueling.
- Every user is different. While some enjoy strong, fast engine braking for its sporty and responsive feel, others will find it uncomfortable and rough. Driveability is a very individual aspect of a vehicle's driveability.

Nonetheless, there are some guidelines on how to use this script for optimal results:
- Take measurements. You want to know how low you can go in the RPM range and how high you can go on the intake pressure. Engine braking occurs right before the idle region, so for example: If you engine idles at 1200 RPM and 400 hPa intake pressure, 1700 RPM and 330 hPa are a good starting point for the "full fuel" values.
- Take small steps. After every modification, you should reflash your ECU and take at least one exhaustive test drive. Check idling when the engine is cold and when it's hot, try going sporty in low gears and touristic in high gears. See if your bike "feels right" for you. If it doesn't: make small adjustments and repeat. It took me ten iterations to get a map that combines good driveability, strong engine braking and a decent amount of noise.
- Once you found good values for full fuel RPM and pressure, experiment with the smoothing parameters.
  - If the smoothing region is too large, engine braking will be reduced as the engine operates less of the time in the zeroed region. Also popping will increase as low fuel values occur more frequently.
  - If the smoothing region is too small, load changes will feel rough. The bike may feel "jerky" and less controllable as more movement is introduced into the drivetrain. Popping will decrease and eventually disappear.
- As for popping, there are two extremes:
  - Maximum smoothing: There are no zeroes in the engine braking region as smoothing has increased all values. The engine will pop a lot and become very loud.
  - No smoothing: There are zeroes in the engine braking region and a direct transition to full fueling on the given RPM and pressure limits. No popping will occur as the engine is either fuelled normally or gets no fuel at all. 

## Known Issues
No error checking is performed.

## TODO