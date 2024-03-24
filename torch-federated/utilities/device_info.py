import socket
import psutil
import platform
import GPUtil as GPU


class DeviceInfo:
    def __init__(self):
        self.ip_address = get_ip_address()
        self.hostname = socket.getfqdn(self.ip_address)
        self.os_type = platform.system()
        self.os_version = platform.version()
        self.os_release = platform.release()
        self.processor = platform.processor()
        self.ram_total, self.ram_available, self.ram_used = get_ram_info()
        self.gpu_info = get_gpu_info()

    def get_ip_address(self):
        return self.ip_address

    def get_hostname(self):
        return self.hostname

    def get_os_type(self):
        return self.os_type

    def get_os_version(self):
        return self.os_version
    
    def get_os_release(self):
        return self.os_release

    def get_processor(self):
        return self.processor
    
    def get_total_ram(self):
        return self.ram_total
    
    def get_available_ram(self):
        return self.ram_available
    
    def get_used_ram(self):
        return self.ram_used
    
    def get_gpu_info(self):
        return self.gpu_info
    


def get_ip_address():
    # Get the IP address of the current device
    return socket.gethostbyname(socket.gethostname())


def get_ram_info():
    # Get RAM information
    ram = psutil.virtual_memory()
    total_ram = ram.total // (1024 ** 3)  # Convert bytes to gigabytes
    available_ram = ram.available // (1024 ** 3)  # Convert bytes to gigabytes
    used_ram = ram.used // (1024 ** 3)  # Convert bytes to gigabytes
    return total_ram, available_ram, used_ram


def get_gpu_info():
    # Get GPU information
    gpus = GPU.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            "id": gpu.id,
            "name": gpu.name,
            "memory.total": f"{gpu.memoryTotal} MB",
            "memory.used": f"{gpu.memoryUsed} MB",
            "memory.free": f"{gpu.memoryFree} MB",
            "temperature": f"{gpu.temperature} °C",
            "load": f"{gpu.load} %"
        })
    return gpu_info


def get_os_info():
    # Get OS information
    os_type = platform.system()
    os_release = platform.release()
    os_version = platform.version()
    return os_type, os_release, os_version


# Example usage
if __name__ == "__main__":
    # Create a DeviceManager instance with an IP address
    device = DeviceInfo()

    # Get information about the device
    print("IP Address:", device.get_ip_address())
    print("Hostname:", device.get_hostname())
    print("OS Type:", device.get_os_type())
    print("OS Version:", device.get_os_version())
    print("OS Release:", device.get_os_release())
    print("Processor:", device.get_processor())
    print("RAM Total:", device.get_total_ram())
    print("RAM Available:", device.get_available_ram())
    print("RAM Used:", device.get_used_ram())
    print("GPU Info:", device.get_gpu_info())
