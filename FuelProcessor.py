from enum import Enum

from pandas import DataFrame


class FuelProcessor:
    class SmoothingAlgorithm(Enum):
        Relative = 1
        RelativeLinear = 2
        Linear = 3

    def __init__(self, full_fuel_rpm: int, full_fuel_press: int, smoothing_factor: float,
                 pressure_weight: float, smoothing_algorithm: SmoothingAlgorithm):
        """
        Instantiates a data container class for processing parameters.
        :param full_fuel_rpm: The maximum engine speed that will not be modified, given in RPM.
        :param full_fuel_press: The maximum intake pressure that will not be modified, given in hPa.
        :param smoothing_factor: Dimensionless factor for edge smoothing. Value dependent on the smoothing
        algorithm.
        :param pressure_weight: A fractional weighting factor between 0 (smooth only along rows/RPM) and 1
        (smooth only along columns/pressure).
        :param smoothing_algorithm: The smoothing algorithm to use.
        """
        self.full_fuel_rpm = full_fuel_rpm
        self.full_fuel_press = full_fuel_press
        self.smoothing_factor = smoothing_factor
        self.pressure_weight = pressure_weight
        self.smoothing_algorithm = smoothing_algorithm

    def smoothing_relative(self, input_data: DataFrame, max_row: int, max_col: int):
        """
        Smooths a zeroed region using relative interpolation.
        :param input_data: The fuel table to be smoothed
        :param max_row: The maximum exclusive row to be processed
        :param max_col: The maximum exclusive column to be processed
        :return: Nothing (in place modification of input_data)
        """
        weight_pressure = self.pressure_weight
        weight_rpm = (1 - self.pressure_weight)

        for row in range(max_row - 1, 0, -1):
            for col in range(max_col - 1, 1, -1):  # omit the zero column
                # get neighbours
                neighbour_press = input_data.iloc[row, col + 1]
                neighbour_rpm = input_data.iloc[row + 1, col]

                # calculate value of change
                val_rpm = 0 if weight_rpm == 0 else (
                        (neighbour_rpm / self.smoothing_factor) * weight_rpm)
                val_press = 0 if weight_pressure == 0 else (
                        (neighbour_press / self.smoothing_factor) * weight_pressure)

                # apply weighted smoothing
                fuel_value = (val_press + val_rpm) / 2
                if fuel_value < 0:
                    fuel_value = 0

                # round to full int and store
                input_data.iloc[row, col] = int(fuel_value)

    def smoothing_relative_linear(self, input_data: DataFrame, max_row: int, max_col: int):
        """
            Smooths a zeroed region using both relative and linear interpolation.
            :param input_data: The fuel table to be smoothed
            :param max_row: The maximum exclusive row to be processed
            :param max_col: The maximum exclusive column to be processed
            :return: Nothing (in place modification of input_data)
            """
        smoothing_factor_pressure = self.smoothing_factor * self.pressure_weight
        smoothing_factor_rpm = self.smoothing_factor * (1 - self.pressure_weight)

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

    def smoothing_linear(self, input_data: DataFrame, max_row: int, max_col: int):
        """
            Smooths a zeroed region using linear interpolation.
            :param input_data: The fuel table to be smoothed
            :param max_row: The maximum exclusive row to be processed
            :param max_col: The maximum exclusive column to be processed
            :return: Nothing (in place modification of input_data)
            """
        smoothing_factor_pressure = self.smoothing_factor * self.pressure_weight
        smoothing_factor_rpm = self.smoothing_factor * (1 - self.pressure_weight)

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

    def find_limits(self, input_data: DataFrame):
        """
        Finds the row and column indices within given RPM and pressure limits.
        :param input_data: The input fuel table.
        :return: The maximum exclusive row and column to be processed.
        """

        # find lowest row
        max_row = 0  # exclusive
        for i in range(1, len(input_data)):
            curr_rpm = input_data[0][i]
            if curr_rpm >= self.full_fuel_rpm:
                max_row = i
            else:
                break

        max_col = 0  # exclusive
        for i in range(1, len(input_data.columns)):
            curr_press = input_data[i][0]
            if curr_press < self.full_fuel_press:
                max_col = i
            else:
                break

        return max_row, max_col

    def process_zero_and_smooth(self, input_data: DataFrame):
        """
        Zeroes a defined region of a fuel table, then applies the selected smoothing function.
        :param input_data: The fuel table to be processed
        :return: Nothing (in place modification of input_data)
        """
        max_row, max_col = self.find_limits(input_data)

        # insert zeroes
        input_data.iloc[1:max_row, 1:max_col] = 0

        # smoothing
        if self.smoothing_algorithm is self.SmoothingAlgorithm.Relative:
            self.smoothing_relative(input_data, max_row, max_col)
        elif self.smoothing_algorithm is self.SmoothingAlgorithm.RelativeLinear:
            self.smoothing_relative_linear(input_data, max_row, max_col)
        elif self.smoothing_algorithm is self.SmoothingAlgorithm.Linear:
            self.smoothing_linear(input_data, max_row, max_col)
        else:
            pass

    def process_multiply(self, input_data: DataFrame):
        """
        EXPERIMENTAL
        :param input_data: The fuel table to be processed
        :return: Nothing (in place modification of input_data)
            """
        max_row, max_col = self.find_limits(input_data)
        weight_pressure = self.pressure_weight
        weight_rpm = (1 - self.pressure_weight)

        for row in range(max_row - 1, 0, -1):
            for col in range(max_col - 1, 0, -1):
                fuel_input = input_data.iloc[row, col]

                # get magnitude of change
                diff_press = input_data.iloc[0, max_col] - input_data.iloc[0, col]
                diff_rpm = input_data.iloc[row, 0] - input_data.iloc[max_row, 0]

                val_press = fuel_input / (self.smoothing_factor * diff_press)
                val_rpm = fuel_input / (self.smoothing_factor * diff_rpm)

                fuel_value = (val_press * weight_pressure) + (weight_rpm * val_rpm)

                input_data.iloc[row, col] = int(fuel_value)
