import platform
import psutil

def get_system_info():
    return {
        "os_name":        platform.system(),
        "os_version":     platform.release(),
        "cpu_model":      platform.processor() or "unknown",
        "physical_cores": psutil.cpu_count(logical=False) or 1,
        "logical_cores":  psutil.cpu_count(logical=True)  or 1,
        "ram_gb":         round(psutil.virtual_memory().total / (1024**3), 1),
    }
