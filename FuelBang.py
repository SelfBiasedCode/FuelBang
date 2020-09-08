from enum import Enum
import pandas


class Smoothing(Enum):
    Relative = 1
    RelativeLinear = 2
    Linear = 3


class BangData:
    def __init__(self, full_fuel_rpm=1700, full_fuel_press=330, smoothing_factor=5.0,
                 pressure_weight=0.5, smoothing_mode=Smoothing.RelativeLinear):
        self.full_fuel_rpm = full_fuel_rpm
        self.full_fuel_press = full_fuel_press
        self.smoothing_factor = smoothing_factor  # reduction per delta
        self.pressure_weight = pressure_weight  # 0: smooth only over RPM, 1: smooth only over hPa
        self.smoothing_mode = smoothing_mode


def smoothing_relative(input_data, max_row, max_col, bang_data):
    weight_pressure = bang_data.pressure_weight
    weight_rpm = (1 - bang_data.pressure_weight)

    for row in range(max_row - 1, 0, -1):
        for col in range(max_col - 1, 1, -1):  # omit the zero column
            # get neighbours
            neighbour_press = input_data.iloc[row, col + 1]
            neighbour_rpm = input_data.iloc[row + 1, col]

            # calculate value of change
            val_rpm = 0 if weight_rpm == 0 else ((neighbour_rpm / bang_data.smoothing_factor) * weight_rpm)
            val_press = 0 if weight_pressure == 0 else (
                        (neighbour_press / bang_data.smoothing_factor) * weight_pressure)

            # apply weighted smoothing
            fuel_value = (val_press + val_rpm) / 2
            if fuel_value < 0:
                fuel_value = 0

            # round to full int and store
            input_data.iloc[row, col] = int(fuel_value)


def smoothing_relative_linear(input_data, max_row, max_col, bang_data):
    smoothing_factor_pressure = bang_data.smoothing_factor * bang_data.pressure_weight
    smoothing_factor_rpm = bang_data.smoothing_factor * (1 - bang_data.pressure_weight)

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


def smoothing_linear(input_data, max_row, max_col, bang_data):
    smoothing_factor_pressure = bang_data.smoothing_factor * bang_data.pressure_weight
    smoothing_factor_rpm = bang_data.smoothing_factor * (1 - bang_data.pressure_weight)

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


def sanitize_input(data_frame):
    # TuneECU exports miss the first element, so we have to insert it manually
    data_frame[0][0] = 0


def desanitize_input(data_frame):
    # delete the previously added element to prevent import problems with TuneECU
    data_frame[0][0] = pandas.np.NaN


def find_limits(input_data, bang_data):
    # find lowest row
    max_row = 0  # exclusive
    for i in range(1, len(input_data)):
        curr_rpm = input_data[0][i]
        if curr_rpm >= bang_data.full_fuel_rpm:
            max_row = i
        else:
            break

    max_col = 0  # exclusive
    for i in range(1, len(input_data.columns)):
        curr_press = input_data[i][0]
        if curr_press < bang_data.full_fuel_press:
            max_col = i
        else:
            break

    return max_row, max_col


def process_zero_and_smooth(input_data, bang_data):
    max_row, max_col = find_limits(input_data, bang_data)

    # insert zeroes
    input_data.iloc[1:max_row, 1:max_col] = 0

    # smoothing
    if bang_data.smoothing_mode is Smoothing.Relative:
        smoothing_relative(input_data, max_row, max_col, bang_data)
    elif bang_data.smoothing_mode is Smoothing.RelativeLinear:
        smoothing_relative_linear(input_data, max_row, max_col, bang_data)
    elif bang_data.smoothing_mode is Smoothing.Linear:
        smoothing_linear(input_data, max_row, max_col, bang_data)
    else:
        pass


def process_multiply(input_data, bang_data):
    max_row, max_col = find_limits(input_data, bang_data)
    weight_pressure = bang_data.pressure_weight
    weight_rpm = (1 - bang_data.pressure_weight)

    for row in range(max_row - 1, 0, -1):
        for col in range(max_col - 1, 0, -1):
            fuel_input = input_data.iloc[row, col]

            # get magnitude of change
            diff_press = input_data.iloc[0, max_col] - input_data.iloc[0, col]
            diff_rpm = input_data.iloc[row, 0] - input_data.iloc[max_row, 0]

            val_press = fuel_input / (bang_data.smoothing_factor * diff_press)
            val_rpm = fuel_input / (bang_data.smoothing_factor * diff_rpm)

            fuel_value = (val_press * weight_pressure) + (weight_rpm * val_rpm)

            input_data.iloc[row, col] = int(fuel_value)


def difference(current, orig, relative):
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


def main():
    input_path_l1 = r"C:\path\to\file"
    input_path_l2 = r"C:\path\to\file"
    input_path_prev_comp = r"C:\path\to\file"

    output_path_l1 = r"C:\path\to\file"
    output_path_l2 = r"C:\path\to\file"
    output_path_l1_comp_base_abs = r"C:\path\to\file"
    output_path_l1_comp_prev_abs = r"C:\path\to\file"
    output_path_l1_comp_base_rel = r"C:\path\to\file"
    output_path_l1_comp_prev_rel = r"C:\path\to\file"

    # read data
    data_l1 = pandas.read_csv(input_path_l1, header=None, delimiter='\t', dtype="Int64")
    data_l2 = pandas.read_csv(input_path_l2, header=None, delimiter='\t', dtype="Int64")
    data_l1_baseline = pandas.read_csv(input_path_l1, header=None, delimiter='\t', dtype="Int64")
    data_l1_prev = pandas.read_csv(input_path_prev_comp, header=None, delimiter='\t', dtype="Int64")

    # preprocess
    sanitize_input(data_l1)
    sanitize_input(data_l2)
    sanitize_input(data_l1_prev)

    bang_data1_1 = BangData(full_fuel_rpm=1700, full_fuel_press=330, smoothing_factor=7,
                            pressure_weight=0.8, smoothing_mode=Smoothing.RelativeLinear)

    # process
    process_zero_and_smooth(data_l1, bang_data1_1)
    process_zero_and_smooth(data_l2, bang_data1_1)

    # calculate absolute and relative differences
    diff_baseline_abs = difference(data_l1, data_l1_baseline, False)
    diff_prev_abs = difference(data_l1, data_l1_prev, False)
    diff_baseline_rel = difference(data_l1, data_l1_baseline, True)
    diff_prev_rel = difference(data_l1, data_l1_prev, True)

    # posstprocess data to make it importable
    desanitize_input(data_l1)
    desanitize_input(data_l2)

    # save data
    data_l1.to_csv(output_path_l1, header=False, index=False, sep='\t')
    data_l2.to_csv(output_path_l2, header=False, index=False, sep='\t')
    diff_baseline_abs.to_csv(output_path_l1_comp_base_abs, header=False, index=False, sep='\t')
    diff_prev_abs.to_csv(output_path_l1_comp_prev_abs, header=False, index=False, sep='\t')
    diff_baseline_rel.to_csv(output_path_l1_comp_base_rel, header=False, index=False, sep='\t')
    diff_prev_rel.to_csv(output_path_l1_comp_prev_rel, header=False, index=False, sep='\t')


if __name__ == '__main__':
    main()
