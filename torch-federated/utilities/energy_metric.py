import pyRAPL

pyRAPL.setup()
measure = pyRAPL.Measurement('bar')
measure.begin()

# ...
# Instructions to be evaluated.
# ...

measure.end()

print(measure.result)