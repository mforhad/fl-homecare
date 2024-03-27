import subprocess
import time

from data.dataloader import fl_config

def get_gpu_power():
    command = "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits"
    output = subprocess.check_output(command, shell=True)

    power = float(output.strip())
    print(f"GPU power used: {power} W")

    return float(output.strip())

def get_gpu_energy_consumption(time_elapsed):
    if not fl_config.should_use_gpu:
        return 0

    try:
        return get_gpu_power() * time_elapsed
    except Exception as e:
        print("No GPU used!")
        return 0

def main():
    start_time = time.time()

    while True:
        time_elapsed = time.time() - start_time
        energy = get_gpu_energy_consumption(time_elapsed)
        print(f"Energy Consumption: {energy} Joules")

        time.sleep(1)  # Update every 1 second

if __name__ == "__main__":
    main()
