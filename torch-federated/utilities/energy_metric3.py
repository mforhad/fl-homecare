import pyRAPL
pyRAPL.setup()

report = pyRAPL.outputs.DataFrameOutput()

with pyRAPL.Measurement('bar',output=report):
    for _ in range(1000000):
        pass

report.data.head()