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

from os import makedirs
from os import path

import pandas
from docopt import docopt
from pandas import DataFrame

from FuelProcessor import FuelProcessor


class FuelBang:
    # # # Internal Classes
    class RunParameters:

        def __init__(self, input_path_l1: str, input_path_l2: str, input_path_l1_prev: str, input_path_l1_base: str,
                     output_dir: str, fuel_processor: FuelProcessor):
            self.input_path_l1 = input_path_l1
            self.input_path_l2 = input_path_l2
            self.input_path_l1_prev = input_path_l1_prev
            self.input_path_l1_base = input_path_l1_base
            self.output_dir = output_dir
            self.fuel_processor = fuel_processor

    # # # Functions

    def __init__(self, params: RunParameters):
        self.params = params

    def sanitize_input(self, input_data: DataFrame):
        """
        Adds am element at 0,0 to a data frame to fix a TuneECU export problem
        :param input_data: The data frame to be sanitized.
        :return: Nothing (in place modification of data_frame).
        """
        input_data[0][0] = 0

    def desanitize_input(self, input_data: DataFrame):
        """
        Deletes the element at 0,0 to prevent import problems with TuneECU
        :param input_data: The data frame to be desanitized.
        :return: Nothing (in place modification of data_frame).
        """
        input_data[0][0] = pandas.np.NaN

    def difference(self, current: DataFrame, orig: DataFrame, relative: bool):
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

    def run(self):
        # recursively create output path if required
        makedirs(self.params.output_dir, exist_ok=True)

        # read data
        data_l1 = pandas.read_csv(self.params.input_path_l1, header=None, delimiter='\t', dtype="Int64")
        data_l2 = pandas.read_csv(self.params.input_path_l2, header=None, delimiter='\t', dtype="Int64")
        data_l1_prev = pandas.read_csv(self.params.input_path_l1_prev, header=None, delimiter='\t', dtype="Int64")
        data_l1_baseline = pandas.read_csv(self.params.input_path_l1_base, header=None, delimiter='\t',
                                           dtype="Int64")

        # preprocess
        self.sanitize_input(data_l1)
        self.sanitize_input(data_l2)
        self.sanitize_input(data_l1_prev)
        self.sanitize_input(data_l1_baseline)

        # process
        self.params.fuel_processor.process_zero_and_smooth(data_l1)
        self.params.fuel_processor.process_zero_and_smooth(data_l2)

        # calculate absolute and relative differences
        diff_prev_abs = self.difference(data_l1, data_l1_prev, False)
        diff_prev_rel = self.difference(data_l1, data_l1_prev, True)
        diff_base_abs = self.difference(data_l1, data_l1_baseline, False)
        diff_base_rel = self.difference(data_l1, data_l1_baseline, True)

        # postprocess data to make it importable
        self.desanitize_input(data_l1)
        self.desanitize_input(data_l2)

        # save data
        data_l1.to_csv(path.join(self.params.output_dir, "FuelBang_L1"), header=False, index=False, sep='\t')
        data_l2.to_csv(path.join(self.params.output_dir, "FuelBang_L2"), header=False, index=False, sep='\t')
        diff_prev_abs.to_csv(path.join(self.params.output_dir, "FuelBang_Prev_Abs_L1"), header=False, index=False,
                             sep='\t')
        diff_prev_rel.to_csv(path.join(self.params.output_dir, "FuelBang_Prev_Rel_L1"), header=False, index=False,
                             sep='\t')
        diff_base_abs.to_csv(path.join(self.params.output_dir, "FuelBang_Base_Abs_L1"), header=False, index=False,
                             sep='\t')
        diff_base_rel.to_csv(path.join(self.params.output_dir, "FuelBang_Base_Rel_L1"), header=False, index=False,
                             sep='\t')


def get_params():
    args = docopt(__doc__, version='1.0.0b')

    # detect algorithm
    if args["--linear"]:
        algorithm = FuelProcessor.SmoothingAlgorithm.Linear
    elif args["--relative"]:
        algorithm = FuelProcessor.SmoothingAlgorithm.Relative
    elif args["--relativeLinear"]:
        algorithm = FuelProcessor.SmoothingAlgorithm.RelativeLinear
    else:
        raise ValueError("Unknown or no smoothing algorithm selected!")

    # if no base path was given, use L1 input table
    if args["--L1_base"] is None:
        base_path = args["--L1"]
    else:
        base_path = args["--L1_base"]

    # assemble processing parameters
    fuel_processor = FuelProcessor(full_fuel_rpm=int(args["--ff_rpm"]),
                                   full_fuel_press=int(args["--ff_press"]),
                                   smoothing_factor=float(args["--sf"]),
                                   pressure_weight=float(args["--pw"]),
                                   smoothing_algorithm=algorithm)

    # assemble and return full parameters
    result = FuelBang.RunParameters(input_path_l1=args["--L1"], input_path_l2=args["--L2"],
                                    input_path_l1_prev=args["--L1_prev"],
                                    input_path_l1_base=base_path, output_dir=args["--out"],
                                    fuel_processor=fuel_processor)
    return result


if __name__ == '__main__':
    parameters = get_params()
    banger = FuelBang(parameters)
    banger.run()
