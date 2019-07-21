import os
import json
import numpy as np
from .pynvml import *

#you can set DFL_TF_MIN_REQ_CAP manually for your build
#the reason why we cannot check tensorflow.version is it requires import tensorflow
tf_min_req_cap = int(os.environ.get("DFL_TF_MIN_REQ_CAP", 35))

class device:
    backend = None
    class Config():
        force_gpu_idx = -1
        multi_gpu = False
        force_gpu_idxs = None
        choose_worst_gpu = False
        gpu_idxs = []
        gpu_names = []
        gpu_compute_caps = []
        gpu_vram_gb = []
        allow_growth = True
        use_fp16 = False
        cpu_only = False
        backend = None
        def __init__ (self, force_gpu_idx = -1,
                            multi_gpu = False,
                            force_gpu_idxs = None,
                            choose_worst_gpu = False,
                            allow_growth = True,
                            use_fp16 = False,
                            cpu_only = False,
                            **in_options):

            self.backend = device.backend
            self.use_fp16 = use_fp16
            self.cpu_only = cpu_only

            if not self.cpu_only:
                self.cpu_only = (self.backend == "tensorflow-cpu")

            if not self.cpu_only:
                self.force_gpu_idx = force_gpu_idx
                self.multi_gpu = multi_gpu
                self.force_gpu_idxs = force_gpu_idxs
                self.choose_worst_gpu = choose_worst_gpu
                self.allow_growth = allow_growth

                self.gpu_idxs = []

                if force_gpu_idxs is not None:
                    for idx in force_gpu_idxs.split(','):
                        idx = int(idx)
                        if device.isValidDeviceIdx(idx):
                            self.gpu_idxs.append(idx)
                else:
                    gpu_idx = force_gpu_idx if (force_gpu_idx >= 0 and device.isValidDeviceIdx(force_gpu_idx)) else device.getBestValidDeviceIdx() if not choose_worst_gpu else device.getWorstValidDeviceIdx()
                    if gpu_idx != -1:
                        if self.multi_gpu:
                            self.gpu_idxs = device.getDeviceIdxsEqualModel( gpu_idx )
                            if len(self.gpu_idxs) <= 1:
                                self.multi_gpu = False
                        else:
                            self.gpu_idxs = [gpu_idx]

                self.cpu_only = (len(self.gpu_idxs) == 0)


            if not self.cpu_only:
                self.gpu_names = []
                self.gpu_compute_caps = []
                self.gpu_vram_gb = []
                for gpu_idx in self.gpu_idxs:
                    self.gpu_names += [device.getDeviceName(gpu_idx)]
                    self.gpu_compute_caps += [ device.getDeviceComputeCapability(gpu_idx) ]
                    self.gpu_vram_gb += [ device.getDeviceVRAMTotalGb(gpu_idx) ]
                self.cpu_only = (len(self.gpu_idxs) == 0)
            else:
                self.gpu_names = ['CPU']
                self.gpu_compute_caps = [99]
                self.gpu_vram_gb = [0]

            if self.cpu_only:
                self.backend = "tensorflow-cpu"

    @staticmethod
    def getValidDeviceIdxsEnumerator():
        if device.backend == "plaidML":
            for i in range(plaidML_devices_count):
                yield i
        elif device.backend == "tensorflow":
            for gpu_idx in range(nvmlDeviceGetCount()):
                cap = device.getDeviceComputeCapability (gpu_idx)
                if cap >= tf_min_req_cap:
                    yield gpu_idx
        elif device.backend == "tensorflow-generic":
            yield 0


    @staticmethod
    def getValidDevicesWithAtLeastTotalMemoryGB(totalmemsize_gb):
        result = []
        if device.backend == "plaidML":
            for i in device.getValidDeviceIdxsEnumerator():
                if plaidML_devices[i]['globalMemSize'] >= totalmemsize_gb*1024*1024*1024:
                     result.append (i)
        elif device.backend == "tensorflow":
            for i in device.getValidDeviceIdxsEnumerator():
                handle = nvmlDeviceGetHandleByIndex(i)
                memInfo = nvmlDeviceGetMemoryInfo( handle )
                if (memInfo.total) >= totalmemsize_gb*1024*1024*1024:
                    result.append (i)
        elif device.backend == "tensorflow-generic":
            return [0]

        return result

    @staticmethod
    def getAllDevicesIdxsList():
        if device.backend == "plaidML":
            return [ *range(plaidML_devices_count) ]
        elif device.backend == "tensorflow":
            return [ *range(nvmlDeviceGetCount() ) ]
        elif device.backend == "tensorflow-generic":
            return [0]

    @staticmethod
    def getValidDevicesIdxsWithNamesList():
        if device.backend == "plaidML":
            return [ (i, plaidML_devices[i]['description'] ) for i in device.getValidDeviceIdxsEnumerator() ]
        elif device.backend == "tensorflow":
            return [ (i, nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)).decode() ) for i in device.getValidDeviceIdxsEnumerator() ]
        elif device.backend == "tensorflow-cpu":
            return [ (0, 'CPU') ]
        elif device.backend == "tensorflow-generic":
            return [ (0, device.getDeviceName(0) ) ]

    @staticmethod
    def getDeviceVRAMTotalGb (idx):
        if device.backend == "plaidML":
            if idx < plaidML_devices_count:
                return plaidML_devices[idx]['globalMemSize'] / (1024*1024*1024)
        elif device.backend == "tensorflow":
            if idx < nvmlDeviceGetCount():
                memInfo = nvmlDeviceGetMemoryInfo(  nvmlDeviceGetHandleByIndex(idx) )
                return round ( memInfo.total / (1024*1024*1024) )

            return 0
        elif device.backend == "tensorflow-generic":
            return 2

    @staticmethod
    def getBestValidDeviceIdx():
        if device.backend == "plaidML":
            idx = -1
            idx_mem = 0
            for i in device.getValidDeviceIdxsEnumerator():
                total = plaidML_devices[i]['globalMemSize']
                if total > idx_mem:
                    idx = i
                    idx_mem = total

            return idx
        elif device.backend == "tensorflow":
            idx = -1
            idx_mem = 0
            for i in device.getValidDeviceIdxsEnumerator():
                memInfo = nvmlDeviceGetMemoryInfo( nvmlDeviceGetHandleByIndex(i) )
                if memInfo.total > idx_mem:
                    idx = i
                    idx_mem = memInfo.total

            return idx
        elif device.backend == "tensorflow-generic":
            return 0

    @staticmethod
    def getWorstValidDeviceIdx():
        if device.backend == "plaidML":
            idx = -1
            idx_mem = sys.maxsize
            for i in device.getValidDeviceIdxsEnumerator():
                total = plaidML_devices[i]['globalMemSize']
                if total < idx_mem:
                    idx = i
                    idx_mem = total

            return idx
        elif device.backend == "tensorflow":
            idx = -1
            idx_mem = sys.maxsize
            for i in device.getValidDeviceIdxsEnumerator():
                memInfo = nvmlDeviceGetMemoryInfo( nvmlDeviceGetHandleByIndex(i) )
                if memInfo.total < idx_mem:
                    idx = i
                    idx_mem = memInfo.total

            return idx
        elif device.backend == "tensorflow-generic":
            return 0

    @staticmethod
    def isValidDeviceIdx(idx):
        if device.backend == "plaidML":
            return idx in [*device.getValidDeviceIdxsEnumerator()]
        elif device.backend == "tensorflow":
            return idx in [*device.getValidDeviceIdxsEnumerator()]
        elif device.backend == "tensorflow-generic":
            return (idx == 0)

    @staticmethod
    def getDeviceIdxsEqualModel(idx):
        if device.backend == "plaidML":
            result = []
            idx_name = plaidML_devices[idx]['description']
            for i in device.getValidDeviceIdxsEnumerator():
                if plaidML_devices[i]['description'] == idx_name:
                    result.append (i)

            return result
        elif device.backend == "tensorflow":
            result = []
            idx_name = nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(idx)).decode()
            for i in device.getValidDeviceIdxsEnumerator():
                if nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)).decode() == idx_name:
                    result.append (i)

            return result
        elif device.backend == "tensorflow-generic":
            return [0] if idx == 0 else []

    @staticmethod
    def getDeviceName (idx):
        if device.backend == "plaidML":
            if idx < plaidML_devices_count:
                return plaidML_devices[idx]['description']
        elif device.backend == "tensorflow":
            if idx < nvmlDeviceGetCount():
                return nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(idx)).decode()
        elif device.backend == "tensorflow-generic":
            if idx == 0:
                return "Generic GeForce GPU"

        return None

    @staticmethod
    def getDeviceID (idx):
        if device.backend == "plaidML":
            if idx < plaidML_devices_count:
                return plaidML_devices[idx]['id'].decode()

        return None

    @staticmethod
    def getDeviceComputeCapability(idx):
        result = 0
        if device.backend == "plaidML":
            return 99
        elif device.backend == "tensorflow":
            if idx < nvmlDeviceGetCount():
                result = nvmlDeviceGetCudaComputeCapability(nvmlDeviceGetHandleByIndex(idx))
        elif device.backend == "tensorflow-generic":
            return 99 if idx == 0 else 0

        return result[0] * 10 + result[1]


force_plaidML = os.environ.get("DFL_FORCE_PLAIDML", "0") == "1" #for OpenCL build , forcing using plaidML even if NVIDIA found
force_tf_cpu = os.environ.get("DFL_FORCE_TF_CPU", "0") == "1"   #for OpenCL build , forcing using tf-cpu if plaidML failed
has_nvml = False
has_nvml_cap = False

#use DFL_FORCE_HAS_NVIDIA_DEVICE=1 if
#- your NVIDIA cannot be seen by OpenCL
#- CUDA build of DFL
has_nvidia_device = os.environ.get("DFL_FORCE_HAS_NVIDIA_DEVICE", "0") == "1"

plaidML_devices = None
def get_plaidML_devices():
    global plaidML_devices
    global has_nvidia_device
    if plaidML_devices is None:
        plaidML_devices = []
        # Using plaidML OpenCL backend to determine system devices and has_nvidia_device
        try:
            os.environ['PLAIDML_EXPERIMENTAL'] = 'false' #this enables work plaidML without run 'plaidml-setup'
            import plaidml
            ctx = plaidml.Context()
            for d in plaidml.devices(ctx, return_all=True)[0]:
                details = json.loads(d.details)
                if details['type'] == 'CPU': #skipping opencl-CPU
                    continue
                if 'nvidia' in details['vendor'].lower():
                    has_nvidia_device = True
                plaidML_devices += [ {'id':d.id,
                                    'globalMemSize' : int(details['globalMemSize']),
                                    'description' : d.description.decode()
                                }]
            ctx.shutdown()
        except:
            pass
    return plaidML_devices

if not has_nvidia_device:
    get_plaidML_devices()

#choosing backend

if device.backend is None and not force_tf_cpu:
    #first trying to load NVSMI and detect CUDA devices for tensorflow backend,
    #even force_plaidML is choosed, because if plaidML will fail, we can choose tensorflow
    try:
        nvmlInit()
        has_nvml = True
        device.backend = "tensorflow"   #set tensorflow backend in order to use device.*device() functions

        gpu_idxs = device.getAllDevicesIdxsList()
        gpu_caps = np.array ( [ device.getDeviceComputeCapability(gpu_idx) for gpu_idx in gpu_idxs ] )

        if len ( np.ndarray.flatten ( np.argwhere (gpu_caps >= tf_min_req_cap) ) ) == 0:
            if not force_plaidML:
                print ("No CUDA devices found with minimum required compute capability: %d.%d. Falling back to OpenCL mode." % (tf_min_req_cap // 10, tf_min_req_cap % 10) )
            device.backend = None
            nvmlShutdown()
        else:
            has_nvml_cap = True
    except:
        #if no NVSMI installed exception will occur
        device.backend = None
        has_nvml = False

if force_plaidML or (device.backend is None and not has_nvidia_device):
    #tensorflow backend was failed without has_nvidia_device , or forcing plaidML, trying to use plaidML backend
    if len(get_plaidML_devices()) == 0:
        #print ("plaidML: No capable OpenCL devices found. Falling back to tensorflow backend.")
        device.backend = None
    else:
        device.backend = "plaidML"
        plaidML_devices_count = len(get_plaidML_devices())

if device.backend is None:
    if force_tf_cpu:
        device.backend = "tensorflow-cpu"
    elif not has_nvml:
        if has_nvidia_device:
            #some notebook systems have NVIDIA card without NVSMI in official drivers
            #in that case considering we have system with one capable GPU and let tensorflow to choose best GPU
            device.backend = "tensorflow-generic"
        else:
            #no NVSMI and no NVIDIA cards, also plaidML was failed, then CPU only
            device.backend = "tensorflow-cpu"
    else:
        if has_nvml_cap:
            #has NVSMI and capable CUDA-devices, but force_plaidML was failed, then we choosing tensorflow
            device.backend = "tensorflow"
        else:
            #has NVSMI, no capable CUDA-devices, also plaidML was failed, then CPU only
            device.backend = "tensorflow-cpu"
