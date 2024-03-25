import pyRAPL

# Initialize pyRAPL
pyRAPL.setup()

# Setup output file
csv_output = pyRAPL.outputs.CSVOutput("utilities/energy_consumption.csv")

# Measure energy consumption of this function
@pyRAPL.measureit
def foo():
    for i in range(1000000):
        pass  # Placeholder, does nothing

# Call the measured function
for _ in range(10):
    foo()

csv_output.save()
