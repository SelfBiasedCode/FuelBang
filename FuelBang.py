"""FuelBang

Usage:
  FuelBang.py --L1=<input_L1> --L2=<input_L2> --L1_prev=<previous_L1> [--L1_base=<base_L1>] --out=<output_directory> [--ff_rpm=<full_fuel_rpm>] [--ff_press=<full_fuel_press>] [--sf=<smoothing_factor>] [--pw=<pressure_weight>] (--linear | --relativeLinear | --relative)

Options:
  --L1=<input_L1>               The path to the L1 table to be modified. Using the stock map is recommended to obtain stable results.
  --L2=<input_L2>               The path to the L2 table to be modified. Using the stock map is recommended to obtain stable results.
  --L1_prev=<previous_L1>       The path to the previous (or a different) L1 table. For comparison only.
  --L1_base=<base_L1>           The path to the base (or a different) L1 table. For comparison only. If omitted, input_L1 will be used.
  --out=<output_directory>      The output directory for all processed results. [default: Processed]
  --ff_rpm=<full_fuel_rpm>      The minimum engine speed to include in the modification range. [default: 1700]
  --ff_press=<full_fuel_press>  The maximum pressure to include in the modification range, given in hPa. [default: 330]
  --sf=<smoothing_factor>       Dimensionless fractional factor for edge smoothing. Value dependent on the smoothing algorithm. [default: 5.0]
  --pw=<pressure_weight>        A fractional weighting factor between 0 (smooth only along rows/RPM) and 1 (smooth only
        along columns/pressure) [default: 0.5]
  --linear                      Use linear smoothing.
  --relative                    Use relative smoothing.
  --relativeLinear              Use relative and linear smoothing.
  -h --help                     Show this screen.
  --version                     Show version.

Minimum Example: FuelBang.py --L1=base_L1.csv --L2=base_L2.csv --L1_prev=mod_L1.csv --out=Processed --linear
Full Example: FuelBang.py --L1=base_L1.csv --L2=base_L2.csv --L1_prev=mod_L1.csv --L1_base=base_L1.csv --out=Processed --ff_rpm=1700 --ff_press=330 --sf=5.0 --pw=0.5 --linear

"""

from enum import Enum
from os import makedirs
from os import path

import pandas
from docopt import docopt
from pandas import DataFrame


def get_params():
    args = docopt(__doc__, version='1.0.0b')

    # detect algorithm
    if args["--linear"]:
        algorithm = Smoothing.Linear
    elif args["--relative"]:
        algorithm = Smoothing.Relative
    elif args["--relativeLinear"]:
        algorithm = Smoothing.RelativeLinear
    else:
        raise ValueError("Unknown or no smoothing algorithm selected!")

    # if no base path was given, use L1 input table
    if args["--L1_base"] is None:
        base_path = args["--L1"]
    else:
        base_path = args["--L1_base"]

    # assemble processing parameters
    process_params = ProcessParams(full_fuel_rpm=int(args["--ff_rpm"]), full_fuel_press=int(args["--ff_press"]),
                                   smoothing_factor=float(args["--sf"]), pressure_weight=float(args["--pw"]),
                                   smoothing_algorithm=algorithm)

    # assemble and return full parameters
    result = Parameters(input_path_l1=args["--L1"], input_path_l2=args["--L2"], input_path_l1_prev=args["--L1_prev"],
                        input_path_l1_base=base_path, output_dir=args["--out"], process_params=process_params)
    return result


class Smoothing(Enum):
    Relative = 1
    RelativeLinear = 2
    Linear = 3


class ProcessParams:
    def __init__(self, full_fuel_rpm: int, full_fuel_press: int, smoothing_factor: float,
                 pressure_weight: float, smoothing_algorithm: Smoothing):
        """
        Instantiates a data container class for processing parameters.
        :param full_fuel_rpm: The maximum engine speed that will not be modified, given in RPM.
        :param full_fuel_press: The maximum intake pressure that will not be modified, given in hPa.
        :param smoothing_factor: Dimensionless factor for edge smoothing. Value dependent on the smoothing algorithm.
        :param pressure_weight: A fractional weighting factor between 0 (smooth only along rows/RPM) and 1 (smooth only
        along columns/pressure).
        :param smoothing_algorithm: The smoothing algorithm to use.
        """
        self.full_fuel_rpm = full_fuel_rpm
        self.full_fuel_press = full_fuel_press
        self.smoothing_factor = smoothing_factor
        self.pressure_weight = pressure_weight
        self.smoothing_algorithm = smoothing_algorithm


def smoothing_relative(input_data: DataFrame, max_row: int, max_col: int, process_params: ProcessParams):
    """
    Smooths a zeroed region using relative interpolation.
    :param input_data: The fuel table to be smoothed
    :param max_row: The maximum exclusive row to be processed
    :param max_col: The maximum exclusive column to be processed
    :param process_params: The smoothing parameters to be used
    :return: Nothing (in place modification of input_data)
    """
    weight_pressure = process_params.pressure_weight
    weight_rpm = (1 - process_params.pressure_weight)

    for row in range(max_row - 1, 0, -1):
        for col in range(max_col - 1, 1, -1):  # omit the zero column
            # get neighbours
            neighbour_press = input_data.iloc[row, col + 1]
            neighbour_rpm = input_data.iloc[row + 1, col]

            # calculate value of change
            val_rpm = 0 if weight_rpm == 0 else ((neighbour_rpm / process_params.smoothing_factor) * weight_rpm)
            val_press = 0 if weight_pressure == 0 else (
                    (neighbour_press / process_params.smoothing_factor) * weight_pressure)

            # apply weighted smoothing
            fuel_value = (val_press + val_rpm) / 2
            if fuel_value < 0:
                fuel_value = 0

            # round to full int and store
            input_data.iloc[row, col] = int(fuel_value)


def smoothing_relative_linear(input_data: DataFrame, max_row: int, max_col: int, process_params: ProcessParams):
    """
        Smooths a zeroed region using both relative and linear interpolation.
        :param input_data: The fuel table to be smoothed
        :param max_row: The maximum exclusive row to be processed
        :param max_col: The maximum exclusive column to be processed
        :param process_params: The smoothing parameters to be used
        :return: Nothing (in place modification of input_data)
        """
    smoothing_factor_pressure = process_params.smoothing_factor * process_params.pressure_weight
    smoothing_factor_rpm = process_params.smoothing_factor * (1 - process_params.pressure_weight)

    for row in range(max_row - 1, 0, -1):
        for col in range(max_col - 1, 1, -1):  # omit the zero column
            # get neighbours
            neighbour_press = input_data.iloc[row, col + 1]
            neighbour_rpm = input_data.iloc[row + 1, col]

            # get magnitude of change
            diff_press = input_data.iloc[0, col + 1] - input_data.iloc[0, col]
            diff_rpm = input_data.iloc[row, 0] - input_data.iloc[row + 1, 0]

            # calculate value of change
            val_rpm = neighbour_rpm - (smoothing_factor_rpm * diff_rpm)
            val_press = neighbour_press - (smoothing_factor_pressure * diff_press)

            # apply weighted smoothing
            fuel_value = (val_press + val_rpm) / 2
            if fuel_value < 0:
                fuel_value = 0

            # round to full int and store
            input_data.iloc[row, col] = int(fuel_value)


def smoothing_linear(input_data: DataFrame, max_row: int, max_col: int, process_params: ProcessParams):
    """
        Smooths a zeroed region using linear interpolation.
        :param input_data: The fuel table to be smoothed
        :param max_row: The maximum exclusive row to be processed
        :param max_col: The maximum exclusive column to be processed
        :param process_params: The smoothing parameters to be used
        :return: Nothing (in place modification of input_data)
        """
    smoothing_factor_pressure = process_params.smoothing_factor * process_params.pressure_weight
    smoothing_factor_rpm = process_params.smoothing_factor * (1 - process_params.pressure_weight)

    for row in range(max_row - 1, 0, -1):
        for col in range(max_col - 1, 1, -1):  # omit the zero column
            # get neighbours
            neighbour_press = input_data.iloc[row, max_col]
            neighbour_rpm = input_data.iloc[max_row, col]

            # get magnitude of change
            diff_press = input_data.iloc[0, max_col] - input_data.iloc[0, col]
            diff_rpm = input_data.iloc[row, 0] - input_data.iloc[max_row, 0]

            # calculate value of change
            val_rpm = neighbour_rpm - (smoothing_factor_rpm * diff_rpm)
            val_press = neighbour_press - (smoothing_factor_pressure * diff_press)

            # apply weighted smoothing
            fuel_value = (val_press + val_rpm) / 2
            if fuel_value < 0:
                fuel_value = 0

            # round to full int and store
            input_data.iloc[row, col] = int(fuel_value)


def sanitize_input(input_data: DataFrame):
    """
    Adds am element at 0,0 to a data frame to fix a TuneECU export problem
    :param input_data: The data frame to be sanitized.
    :return: Nothing (in place modification of data_frame).
    """
    input_data[0][0] = 0


def desanitize_input(input_data: DataFrame):
    """
    Deletes the element at 0,0 to prevent import problems with TuneECU
    :param input_data: The data frame to be desanitized.
    :return: Nothing (in place modification of data_frame).
    """
    input_data[0][0] = pandas.np.NaN


def find_limits(input_data: DataFrame, process_params: ProcessParams):
    """
    Finds the row and column indices within given RPM and pressure limits.
    :param input_data: The input fuel table.
    :param process_params: The processing parameters.
    :return: The maximum exclusive row and column to be processed.
    """

    # find lowest row
    max_row = 0  # exclusive
    for i in range(1, len(input_data)):
        curr_rpm = input_data[0][i]
        if curr_rpm >= process_params.full_fuel_rpm:
            max_row = i
        else:
            break

    max_col = 0  # exclusive
    for i in range(1, len(input_data.columns)):
        curr_press = input_data[i][0]
        if curr_press < process_params.full_fuel_press:
            max_col = i
        else:
            break

    return max_row, max_col


def process_zero_and_smooth(input_data: DataFrame, process_params: ProcessParams):
    """
    Zeroes a defined region of a fuel table, then applies the selected smoothing function.
    :param input_data: The fuel table to be processed
    :param process_params: Processing parameters
    :return: Nothing (in place modification of input_data)
    """
    max_row, max_col = find_limits(input_data, process_params)

    # insert zeroes
    input_data.iloc[1:max_row, 1:max_col] = 0

    # smoothing
    if process_params.smoothing_algorithm is Smoothing.Relative:
        smoothing_relative(input_data, max_row, max_col, process_params)
    elif process_params.smoothing_algorithm is Smoothing.RelativeLinear:
        smoothing_relative_linear(input_data, max_row, max_col, process_params)
    elif process_params.smoothing_algorithm is Smoothing.Linear:
        smoothing_linear(input_data, max_row, max_col, process_params)
    else:
        pass


def process_multiply(input_data: DataFrame, process_params: ProcessParams):
    """
    EXPERIMENTAL
    :param input_data: The fuel table to be processed
    :param process_params: Processing parameters
    :return: Nothing (in place modification of input_data)
        """
    max_row, max_col = find_limits(input_data, process_params)
    weight_pressure = process_params.pressure_weight
    weight_rpm = (1 - process_params.pressure_weight)

    for row in range(max_row - 1, 0, -1):
        for col in range(max_col - 1, 0, -1):
            fuel_input = input_data.iloc[row, col]

            # get magnitude of change
            diff_press = input_data.iloc[0, max_col] - input_data.iloc[0, col]
            diff_rpm = input_data.iloc[row, 0] - input_data.iloc[max_row, 0]

            val_press = fuel_input / (process_params.smoothing_factor * diff_press)
            val_rpm = fuel_input / (process_params.smoothing_factor * diff_rpm)

            fuel_value = (val_press * weight_pressure) + (weight_rpm * val_rpm)

            input_data.iloc[row, col] = int(fuel_value)


def difference(current: DataFrame, orig: DataFrame, relative: bool):
    """
    Caclulates the relative or absolute differences in cell values between two fuel tables.
    :param current: The current fuel table
    :param orig: The original fuel table
    :param relative: If True: calculate relative difference, if False: calculate absolute differences
    :return: A data frame of the same dimensions as current, containing the calculated differences
    """
    result = current.copy()
    for row in range(1, len(result)):
        for col in range(1, len(result.columns) - 1):
            curr_val = current.iloc[row, col]
            orig_val = orig.iloc[row, col]
            if relative:
                if orig_val == 0:
                    if curr_val > 0:
                        result.iloc[row, col] = int(+100)
                    elif curr_val < 0:
                        result.iloc[row, col] = int(-100)
                    else:
                        result.iloc[row, col] = int(0)
                else:
                    result.iloc[row, col] = int(((curr_val - orig_val) / orig_val) * 100)
            else:
                result.iloc[row, col] = int(curr_val - orig_val)

    return result


class Parameters:
    def __init__(self, input_path_l1: str, input_path_l2: str, input_path_l1_prev: str, input_path_l1_base: str, output_dir: str,
                 process_params: ProcessParams):
        self.input_path_l1 = input_path_l1
        self.input_path_l2 = input_path_l2
        self.input_path_l1_prev = input_path_l1_prev
        self.input_path_l1_base = input_path_l1_base
        self.output_dir = output_dir
        self.process_params = process_params


def main(params: Parameters):
    # recursively create output path if required
    makedirs(params.output_dir, exist_ok=True)

    # read data
    data_l1 = pandas.read_csv(params.input_path_l1, header=None, delimiter='\t', dtype="Int64")
    data_l2 = pandas.read_csv(params.input_path_l2, header=None, delimiter='\t', dtype="Int64")
    data_l1_prev = pandas.read_csv(params.input_path_l1_prev, header=None, delimiter='\t', dtype="Int64")
    data_l1_baseline = pandas.read_csv(params.input_path_l1_base, header=None, delimiter='\t', dtype="Int64")

    # preprocess
    sanitize_input(data_l1)
    sanitize_input(data_l2)
    sanitize_input(data_l1_prev)
    sanitize_input(data_l1_baseline)

    # process
    process_zero_and_smooth(data_l1, params.process_params)
    process_zero_and_smooth(data_l2, params.process_params)

    # calculate absolute and relative differences
    diff_prev_abs = difference(data_l1, data_l1_prev, False)
    diff_prev_rel = difference(data_l1, data_l1_prev, True)
    diff_base_abs = difference(data_l1, data_l1_baseline, False)
    diff_base_rel = difference(data_l1, data_l1_baseline, True)

    # postprocess data to make it importable
    desanitize_input(data_l1)
    desanitize_input(data_l2)

    # save data
    data_l1.to_csv(path.join(params.output_dir, "FuelBang_L1"), header=False, index=False, sep='\t')
    data_l2.to_csv(path.join(params.output_dir, "FuelBang_L2"), header=False, index=False, sep='\t')
    diff_prev_abs.to_csv(path.join(params.output_dir, "FuelBang_Prev_Abs_L1"), header=False, index=False, sep='\t')
    diff_prev_rel.to_csv(path.join(params.output_dir, "FuelBang_Prev_Rel_L1"), header=False, index=False, sep='\t')
    diff_base_abs.to_csv(path.join(params.output_dir, "FuelBang_Base_Abs_L1"), header=False, index=False, sep='\t')
    diff_base_rel.to_csv(path.join(params.output_dir, "FuelBang_Base_Rel_L1"), header=False, index=False, sep='\t')


if __name__ == '__main__':
    parameters = get_params()
    main(parameters)
