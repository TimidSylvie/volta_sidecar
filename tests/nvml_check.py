"""Query every GPU through whatever libnvidia-ml.so.1 the loader finds (the proxy, when on LD_LIBRARY_PATH)."""
import ctypes
nv = ctypes.CDLL("libnvidia-ml.so.1")
def ck(rc, what):
    if rc != 0:
        msg = nv.nvmlErrorString; msg.restype = ctypes.c_char_p
        raise SystemExit(f"{what}: NVML error {rc} ({msg(rc).decode()})")
ck(nv.nvmlInit_v2(), "init")
n = ctypes.c_uint(); ck(nv.nvmlDeviceGetCount_v2(ctypes.byref(n)), "count")
print("devices:", n.value)
class Mem(ctypes.Structure): _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
class Util(ctypes.Structure): _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
class Proc(ctypes.Structure): _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong), ("gi", ctypes.c_uint), ("ci", ctypes.c_uint)]
for i in range(n.value):
    d = ctypes.c_void_p(); ck(nv.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(d)), f"handle {i}")
    name = ctypes.create_string_buffer(96); ck(nv.nvmlDeviceGetName(d, name, 96), "name")
    uuid = ctypes.create_string_buffer(96); ck(nv.nvmlDeviceGetUUID(d, uuid, 96), "uuid")
    t = ctypes.c_uint(); ck(nv.nvmlDeviceGetTemperature(d, 0, ctypes.byref(t)), "temp")
    m = Mem(); ck(nv.nvmlDeviceGetMemoryInfo(d, ctypes.byref(m)), "mem")
    u = Util(); ck(nv.nvmlDeviceGetUtilizationRates(d, ctypes.byref(u)), "util")
    p = ctypes.c_uint(); ck(nv.nvmlDeviceGetPowerUsage(d, ctypes.byref(p)), "power")
    lim = ctypes.c_uint(); ck(nv.nvmlDeviceGetEnforcedPowerLimit(d, ctypes.byref(lim)), "limit")
    idx = ctypes.c_uint(); ck(nv.nvmlDeviceGetIndex(d, ctypes.byref(idx)), "index")
    back = ctypes.c_void_p(); ck(nv.nvmlDeviceGetHandleByUUID(uuid, ctypes.byref(back)), "by uuid")
    cnt = ctypes.c_uint(16); procs = (Proc * 16)()
    rc = nv.nvmlDeviceGetComputeRunningProcesses_v3(d, ctypes.byref(cnt), procs)
    plist = [procs[k].pid for k in range(cnt.value)] if rc == 0 else f"rc={rc}"
    print(f"[{i}] index={idx.value} {name.value.decode()} {uuid.value.decode()[:16]}.. {t.value} C, "
          f"{m.used >> 20}/{m.total >> 20} MiB, util {u.gpu}%, {p.value / 1000:.1f} W of {lim.value / 1000:.0f} W, "
          f"uuid lookup {'same handle' if back.value == d.value else 'DIFFERENT'}, processes {plist}")
# A mixed pair should get the same conservative answer as two GPUs without a direct interconnect.
if n.value >= 2:
    a, b, s = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_int()
    nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(a)); nv.nvmlDeviceGetHandleByIndex_v2(n.value - 1, ctypes.byref(b))
    print("cross-driver OnSameBoard rc:", nv.nvmlDeviceOnSameBoard(a, b, ctypes.byref(s)))
ck(nv.nvmlShutdown(), "shutdown"); print("shutdown ok")
