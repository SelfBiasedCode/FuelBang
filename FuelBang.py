from enum import Enum
import pandas

class Smoothing(Enum):
    Relative = 1
    RelativeLinear = 2
    Linear = 3

class BangData:
    def __init__(self, fullFuelRpm=1700, fullFuelPress=330, smoothingFactor=5.0,
                 pressureWeight=0.5, smoothingMode=Smoothing.RelativeLinear):
        self.fullFuelRpm = fullFuelRpm
        self.fullFuelPress = fullFuelPress
        self.smoothingFactor = smoothingFactor # reduction per delta
        self.pressureWeight = pressureWeight # 0: smooth only over RPM, 1: smooth only over hPa
        self.smoothingMode = smoothingMode

def smoothingRelative(inputData, maxRow, maxCol, bangData):
    weightPressure = bangData.pressureWeight
    weightRpm = (1 - bangData.pressureWeight)

    for row in range(maxRow - 1, 0, -1):
        for col in range(maxCol - 1, 1, -1): # omit the zero column
            # get neighbours
            neighbourPress = inputData.iloc[row, col + 1]
            neighbourRpm = inputData.iloc[row + 1, col]

            # calculate value of change
            valRpm = 0 if weightRpm == 0 else ((neighbourRpm / bangData.smoothingFactor) * weightRpm)
            valPress = 0 if weightPressure == 0 else ((neighbourPress / bangData.smoothingFactor )* weightPressure)

            # apply weighted smoothing
            fuelValue = (valPress  + valRpm) / 2
            if (fuelValue < 0):
                fuelValue = 0

            inputData.iloc[row, col] = int(fuelValue)


def smoothingRelativeLinear(inputData, maxRow, maxCol, bangData):
    smoothingFactorPressure = bangData.smoothingFactor * bangData.pressureWeight
    smoothingFactorRpm = bangData.smoothingFactor * (1 - bangData.pressureWeight)

    for row in range(maxRow - 1, 0, -1):
        for col in range(maxCol - 1, 1, -1): # omit the zero column
            # get neighbours
            neighbourPress = inputData.iloc[row, col + 1]
            neighbourRpm = inputData.iloc[row + 1, col]

            # get magnitude of change
            diffPress = inputData.iloc[0, col + 1] - inputData.iloc[0, col]
            diffRpm = inputData.iloc[row, 0] - inputData.iloc[row + 1, 0]

            # calculate value of change
            valRpm = neighbourRpm - (smoothingFactorRpm * diffRpm)
            valPress = neighbourPress - (smoothingFactorPressure * diffPress)

            # apply weighted smoothing
            fuelValue = (valPress  + valRpm) / 2
            if (fuelValue < 0):
                fuelValue = 0

            inputData.iloc[row, col] = int(fuelValue)


def smoothingLinear(inputData, maxRow, maxCol, bangData):
    smoothingFactorPressure = bangData.smoothingFactor * bangData.pressureWeight
    smoothingFactorRpm = bangData.smoothingFactor * (1 - bangData.pressureWeight)

    for row in range(maxRow - 1, 0, -1):
        for col in range(maxCol - 1, 1, -1): # omit the zero column
            # get neighbours
            neighbourPress = inputData.iloc[row, maxCol]
            neighbourRpm = inputData.iloc[maxRow, col]

            # get magnitude of change
            diffPress = inputData.iloc[0, maxCol] - inputData.iloc[0, col]
            diffRpm = inputData.iloc[row, 0] - inputData.iloc[maxRow, 0]

            # calculate value of change
            valRpm = neighbourRpm - (smoothingFactorRpm * diffRpm)
            valPress = neighbourPress - (smoothingFactorPressure * diffPress)

            # apply weighted smoothing
            fuelValue = (valPress + valRpm) / 2
            if (fuelValue < 0):
                fuelValue = 0

            inputData.iloc[row, col] = int(fuelValue)


def sanitizeInput(dataFrame):
    # fix missing element
    dataFrame[0][0] = 0
    # delete empty column
    # del dataFrame[21]

def desanitizeInput(dataFrame):
    # remove  missing element
    dataFrame[0][0] = pandas.np.NaN


def findLimits(inputData, bangData):
    # find lowest row
    maxRow = 0  # exclusive
    for i in range(1, len(inputData)):
        currRpm = inputData[0][i]
        if currRpm >= bangData.fullFuelRpm:
            maxRow = i
        else:
            break

    maxCol = 0  # exclusive
    for i in range(1, len(inputData.columns)):
        currPress = inputData[i][0]
        if currPress < bangData.fullFuelPress:
            maxCol = i
        else:
            break

    return maxRow, maxCol

def processZeroAndSmooth(inputData, bangData):
    maxRow, maxCol = findLimits(inputData, bangData)

    # insert zeroes
    inputData.iloc[1:maxRow, 1:maxCol] = 0

    # smoothing
    if bangData.smoothingMode is Smoothing.Relative:
        smoothingRelative(inputData, maxRow, maxCol, bangData)
    elif bangData.smoothingMode is Smoothing.RelativeLinear:
        smoothingRelativeLinear(inputData, maxRow, maxCol, bangData)
    elif bangData.smoothingMode is Smoothing.Linear:
        smoothingLinear(inputData, maxRow, maxCol, bangData)
    else:
        pass


def processMultiply(inputData, bangData):
    maxRow, maxCol = findLimits(inputData, bangData)
    weightPressure = bangData.pressureWeight
    weightRpm = (1 - bangData.pressureWeight)

    for row in range(maxRow - 1, 0, -1):
        for col in range(maxCol - 1, 0, -1):
            fuelInput = inputData.iloc[row, col]

            # get magnitude of change
            diffPress = inputData.iloc[0, maxCol] - inputData.iloc[0, col]
            diffRpm = inputData.iloc[row, 0] - inputData.iloc[maxRow, 0]

            valPress = fuelInput / (bangData.smoothingFactor * diffPress)
            valRpm = fuelInput / (bangData.smoothingFactor * diffRpm)

            fuelValue = (valPress * weightPressure) + (weightRpm * valRpm)


            inputData.iloc[row, col] = int(fuelValue)

def difference(current, orig, relative):
    result = current.copy()
    for row in range(1, len(result)):
        for col in range(1, len(result.columns)-1):
            currVal = current.iloc[row, col]
            origVal = orig.iloc[row, col]
            if relative:
                if origVal == 0:
                    if currVal > 0:
                        result.iloc[row, col] = int(+100)
                    elif currVal < 0:
                        result.iloc[row, col] = int(-100)
                    else:
                        result.iloc[row, col] = int(0)
                else:
                    result.iloc[row, col] = int(((currVal-origVal) / origVal) * 100)
            else:
                result.iloc[row, col] = int(currVal - origVal)

    return result

def main():
    inputPathL1 = r"C:\path\to\file"
    inputPathL2 = r"C:\path\to\file"
    inputPathPrevComp = r"C:\path\to\file"

    outputPathL1 = r"C:\path\to\file"
    outputPathL2 = r"C:\path\to\file"
    outputPathL1CompBaseAbs = r"C:\path\to\file"
    outputPathL1CompPrevAbs = r"C:\path\to\file"
    outputPathL1CompBaseRel = r"C:\path\to\file"
    outputPathL1CompPrevRel = r"C:\path\to\file"

    dataL1 = pandas.read_csv(inputPathL1, header=None, delimiter='\t', dtype="Int64")
    dataL2 = pandas.read_csv(inputPathL2, header=None, delimiter='\t', dtype="Int64")
    dataL1Baseline = pandas.read_csv(inputPathL1, header=None, delimiter='\t', dtype="Int64")
    dataL1Prev = pandas.read_csv(inputPathPrevComp, header=None, delimiter='\t', dtype="Int64")

    sanitizeInput(dataL1)
    sanitizeInput(dataL2)
    sanitizeInput(dataL1Prev)

    bangData1_1 = BangData(fullFuelRpm=1700, fullFuelPress=330, smoothingFactor=7,
                           pressureWeight=0.8, smoothingMode=Smoothing.RelativeLinear)

    bangData1_2 = BangData(fullFuelRpm=1600, fullFuelPress=330, smoothingFactor=2.3,
                           pressureWeight=0.05, smoothingMode=Smoothing.RelativeLinear)

    bangData2 = BangData(fullFuelRpm=1700, fullFuelPress=330, smoothingFactor=20,
                         pressureWeight=1.0, smoothingMode=Smoothing.Linear)

    bangData3 = BangData(fullFuelRpm=1700, fullFuelPress=330, smoothingFactor=30,
                         pressureWeight=1.0, smoothingMode=Smoothing.RelativeLinear)

    bangData4 = BangData(fullFuelRpm=1600, fullFuelPress=330, smoothingFactor=4,
                         pressureWeight=0.7, smoothingMode=Smoothing.Linear)

    bangData5 = BangData(fullFuelRpm=1700, fullFuelPress=330, smoothingFactor=1,
                         pressureWeight=0.5, smoothingMode=Smoothing.Relative)

    processZeroAndSmooth(dataL1, bangData1_1)
    processZeroAndSmooth(dataL2, bangData1_1)

    diffBaselineAbs = difference(dataL1, dataL1Baseline, False)
    diffPrevAbs = difference(dataL1, dataL1Prev, False)
    diffBaselineRel = difference(dataL1, dataL1Baseline, True)
    diffPrevRel = difference(dataL1, dataL1Prev, True)

    desanitizeInput(dataL1)
    desanitizeInput(dataL2)

    dataL1.to_csv(outputPathL1, header=False, index=False, sep='\t')
    dataL2.to_csv(outputPathL2, header=False, index=False, sep='\t')
    diffBaselineAbs.to_csv(outputPathL1CompBaseAbs, header=False, index=False, sep='\t')
    diffPrevAbs.to_csv(outputPathL1CompPrevAbs, header=False, index=False, sep='\t')
    diffBaselineRel.to_csv(outputPathL1CompBaseRel, header=False, index=False, sep='\t')
    diffPrevRel.to_csv(outputPathL1CompPrevRel, header=False, index=False, sep='\t')

if __name__ == '__main__':
    main()