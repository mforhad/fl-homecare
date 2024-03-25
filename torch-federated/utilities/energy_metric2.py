import pyRAPL

pyRAPL.setup()
meter = pyRAPL.Measurement('bar')
meter.begin()
# ...
for _ in range(1000000):
    pass  # Placeholder, does nothing
# ...
meter.end()

print(meter.result.dram)
print(meter.result.pkg)