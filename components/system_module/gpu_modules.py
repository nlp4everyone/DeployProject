from pynvml import *
import math
nvmlInit()

class GpuModule():
    @staticmethod
    def get_gpu_property(gpu_index = 0):
        # Init object
        h = nvmlDeviceGetHandleByIndex(gpu_index)
        info = nvmlDeviceGetMemoryInfo(h)

        # Total VRAM
        total_vram = info.total
        # Free VRAM
        free_vram = info.free
        # Used VRAM
        used_vram = info.used


        # % Usage
        usage = round((used_vram/total_vram),2)
        return free_vram,total_vram,used_vram,usage

    def _millify(n):
        n = float(n)
        millnames = ['', ' KB', ' MB', ' GB', ' TB']
        millidx = max(0, min(len(millnames) - 1,
                             int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

        result = str(n / 10 ** (3 * millidx))
        # result =
        return str(round(float(result),2)) + str(millnames[millidx])
        # return str(result)
    @staticmethod
    def formmated_property(free_vram,total_vram,used_vram):
        free_vram = GpuModule._millify(free_vram)
        total_vram = GpuModule._millify(total_vram)
        used_vram = GpuModule._millify(used_vram)
        return free_vram,total_vram,used_vram