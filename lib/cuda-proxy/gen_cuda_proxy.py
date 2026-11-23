#!/usr/bin/env python3
"""Turn CUDA's headers and real export table into the proxy's boring per-function glue.

Usage: gen_cuda_proxy.py TYPEDEFS_JSON PROCMAP EXPORTS CUDA_INCLUDE_DIR OUT_DIR
  TYPEDEFS_JSON  parse_typedefs.py output (cudaTypedefs.h: every driver function at every API version)
  PROCMAP        map_procs output: "name version per_thread exported_symbol" (what the real libcuda resolves)
  EXPORTS        every function the real libcuda exports, one per line
Writes OUT_DIR/cp_gen.c (names table + one typed wrapper per typed export) and OUT_DIR/cp_stubs.S (register-
preserving trampolines for the exports without a known signature, and the private-table dispatcher).
"""
import json, pathlib, re, subprocess, sys, tempfile

typedefs_p, procmap_p, exports_p, inc, out = sys.argv[1:6]
out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
exports = sorted(l.strip() for l in open(exports_p) if l.strip())
typedefs = {(t["name"], t["ver"], t["pt"]): t for t in json.load(open(typedefs_p))}
sym_sig = {}
gpa_map = {}
for line in open(procmap_p):
    name, ver, pt, sym = line.split()
    ver, pt = int(ver), int(pt)
    if sym != "-":
        gpa_map[(name, ver, pt)] = sym
        t = typedefs.get((name, ver, pt))
        if t and (sym not in sym_sig or t["ver"] > sym_sig[sym]["ver"]):
            sym_sig[sym] = t

# These calls need enough context that the routing core implements them by hand.
SPECIAL = {"cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetByPCIBusId", "cuCtxGetCurrent", "cuCtxSetCurrent",
           "cuCtxPushCurrent", "cuCtxPushCurrent_v2", "cuCtxPopCurrent", "cuCtxPopCurrent_v2", "cuDeviceCanAccessPeer",
           "cuCtxEnablePeerAccess", "cuCtxDisablePeerAccess", "cuGetProcAddress", "cuGetProcAddress_v2", "cuGetExportTable",
           "cuLibraryLoadData", "cuLibraryLoadFromFile", "cuLibraryUnload", "cuLibraryGetKernel", "cuMemPoolCreate",
           "cuMemCreate", "cuMemGetAllocationGranularity", "cuMemRelease", "cuMemSetAccess", "cuMemGetAccess",
           "cuMemPoolSetAccess", "cuMemPoolGetAccess", "cuMemPrefetchAsync_v2", "cuMemAdvise_v2"}
# These objects exist in both drivers behind one public handle.
PAIRED = {"CUlibrary", "CUkernel", "CUfunction"}      # CUDA accepts a CUkernel where it expects a CUfunction.
HANDLES = {"CUcontext", "CUstream", "CUevent", "CUmodule", "CUfunction", "CUlibrary", "CUkernel", "CUgraph", "CUgraphExec",
           "CUgraphNode", "CUarray", "CUmipmappedArray", "CUmemoryPool", "CUlinkState", "CUexternalMemory",
           "CUexternalSemaphore", "CUgreenCtx", "CUtexref", "CUsurfref", "CUgraphicsResource", "CUuserObject",
           "CUgraphDeviceNode", "CUdevResourceDesc", "CUlogsCallbackHandle", "CUcoredumpCallbackHandle", "CUcheckpointContext"}
DEVICE = {"CUdevice", "CUdevice_v1"}
DPTR64 = {"CUdeviceptr", "CUdeviceptr_v2"}
HOSTMEM_ROUTED = {"cuMemFreeHost", "cuMemHostGetDevicePointer", "cuMemHostGetDevicePointer_v2", "cuMemHostGetFlags",
                  "cuMemHostUnregister"}
HOSTMEM_ALLOC = {"cuMemAllocHost", "cuMemAllocHost_v2", "cuMemHostAlloc"}
HOSTMEM_REGISTER = {"cuMemHostRegister", "cuMemHostRegister_v2"}
HANDLE_RELEASE = {
    "cuArrayDestroy", "cuCoredumpUnregisterCallback", "cuCtxDestroy", "cuCtxDestroy_v2",
    "cuDestroyExternalMemory", "cuDestroyExternalSemaphore", "cuEventDestroy", "cuEventDestroy_v2",
    "cuGraphDestroy", "cuGraphDestroyNode", "cuGraphDeviceNodeDestroy", "cuGraphExecDestroy",
    "cuGraphicsUnregisterResource", "cuGreenCtxDestroy", "cuLinkDestroy", "cuMemPoolDestroy",
    "cuMipmappedArrayDestroy", "cuModuleUnload", "cuStreamDestroy", "cuStreamDestroy_v2",
}
PTR_RELEASE = {"cuIpcCloseMemHandle", "cuMemAddressFree", "cuMemFree", "cuMemFree_v2", "cuMemFreeAsync", "cuMemUnmap"}
COPY_PTR_PAIR = re.compile(r"^cuMemcpy(DtoD(_v2)?|DtoDAsync(_v2)?)?(Async)?(_ptds|_ptsz)?$")
COPY_PEER = re.compile(r"^cuMemcpyPeer(Async)?(_ptds|_ptsz)?$")
CROSS_WAIT_EVENT = re.compile(r"^cuStreamWaitEvent(_ptsz)?$")

def base(t):
    return re.sub(r"\bconst\b", "", t).replace("struct ", "").strip()

# Ask the installed headers for ABI sizes instead of guessing what a typedef means.
types = sorted({p[0] for s in sym_sig.values() for p in s["params"] if not p[2]})
probe = ['#define __CUDA_API_VERSION_INTERNAL 1', '#include <cuda.h>', '#include <cudaTypedefs.h>', '#include <stdio.h>', 'int main(void){']
for i, t in enumerate(types):
    probe.append(f'  {{ {t} v_{i}; (void)v_{i}; printf("%d %zu %d\\n", {i}, sizeof({t}), __builtin_classify_type(v_{i})); }}')
probe.append("return 0;}")
with tempfile.TemporaryDirectory() as td:
    src, exe = pathlib.Path(td) / "p.c", pathlib.Path(td) / "p"
    src.write_text("\n".join(probe))
    r = subprocess.run(["gcc", "-w", "-I", inc, "-o", str(exe), str(src)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("type probe failed:\n" + r.stderr[:3000])
    abi = {}
    for line in subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.splitlines():
        i, size, cls = map(int, line.split())
        abi[types[i]] = (size, cls)

def abi_type(t, is_fp):
    if is_fp or "*" in t:
        return "void *"
    size, cls = abi[t]
    if cls == 5:                          # Pointer-like handle typedefs such as CUcontext.
        return "void *"
    if cls == 8:                          # Floating-point values.
        return "float" if size == 4 else "double"
    if cls == 12 or cls == 13:            # A by-value struct/union needs a hand-written wrapper.
        return None
    return {1: "unsigned char", 2: "unsigned short", 4: "unsigned int", 8: "unsigned long long"}[size]

# Generate one normal C wrapper wherever the signature is known.
idx = {n: i for i, n in enumerate(exports)}
C = ['/* generated by gen_cuda_proxy.py - do not edit */', '#include <stddef.h>',
     'typedef int CUresult;', 'extern void **cp_fn[2]; extern int cp_have[2];',
     'void cp_init(void); int cp_cur(void); void cp_set_cur(int);',
     'void cp_own(const void *, int); void cp_disown(const void *); int cp_owner(const void *, int);',
     'void cp_ptr_own(unsigned long long, int); void cp_ptr_forget(unsigned long long, int); int cp_ptr_owner(unsigned long long);',
     'int cp_dev_side(int, int *); int cp_dev_global(int, int);',
     'void *cp_pair_get(const void *, int); void *cp_pair_pub(const void *, int);',
     'extern int cp_trace_all; void cp_trace(const char *, int, CUresult);',
     'CUresult cp_stage_copy(unsigned long long, int, unsigned long long, int, size_t);',
     'CUresult cp_cross_wait_event(void *, int);',
     '#define NS 801 /* CUDA_ERROR_NOT_SUPPORTED */',
     'const char *const cp_names[] = {'] + [f'    "{n}",' for n in exports] + [
     '};', f'const int cp_count = {len(exports)};',
     'struct cp_gpa_entry { const char *name; int version; unsigned pt; const char *export_name; };',
     'const struct cp_gpa_entry cp_gpa_map[] = {'] + [
         f'    {{"{name}", {ver}, {pt}, "{sym}"}},'
         for (name, ver, pt), sym in sorted(gpa_map.items())
     ] + ['};', f'const int cp_gpa_count = {len(gpa_map)};', '']
typed, untyped = 0, []
for sym in exports:
    if sym in SPECIAL:
        continue
    sig = sym_sig.get(sym)
    if not sig or sig["ret"] != "CUresult":
        untyped.append(sym); continue
    params = sig["params"]
    ats = [abi_type(p[0], p[2]) for p in params]
    if any(a is None for a in ats):
        untyped.append(sym); continue
    names = [f"p{i}" for i in range(len(params))]
    decl = ", ".join(f"{a} {n}" for a, n in zip(ats, names)) or "void"
    body = ["    cp_init();", "    int side = cp_cur();"]
    bases = [base(p[0]) for p in params]
    routed = False
    dev_params = [i for i, b in enumerate(bases) if b in DEVICE]
    if COPY_PTR_PAIR.match(sym) and len(params) >= 3 and ats[0] == ats[1] == "unsigned long long":
        body += ["    int so = cp_ptr_owner(p1), d = cp_ptr_owner(p0);",
                 "    if (so >= 0 && d >= 0 && so != d) return cp_stage_copy(p0, d, p1, so, (size_t)p2);",
                 "    if (d >= 0) side = d; else if (so >= 0) side = so;"]
        routed = True
    elif COPY_PEER.match(sym):
        body += ["    int d = cp_owner(p1, side), so = cp_owner(p3, side);",
                 "    if (so != d) return cp_stage_copy(p0, d, p2, so, (size_t)p4);",
                 "    side = d;"]
        routed = True
    elif CROSS_WAIT_EVENT.match(sym):
        body += ["    int stream_side = cp_owner(p0, side), event_side = cp_owner(p1, side);",
                 "    if (stream_side != event_side) return cp_cross_wait_event(p1, event_side);",
                 "    side = stream_side;"]
        routed = True
    elif sym in HOSTMEM_ROUTED and ats and ats[0] == "void *":
        body += ["    side = cp_owner(p0, side);"]; routed = True
    # The first argument with a known owner picks the driver. Paired handles exist on both sides and cannot decide.
    body.append(f"    int done = {1 if routed else 0}; (void)done;")
    if not routed:
        if sym.startswith("cuLaunchKernelEx"):        # CUlaunchConfig stores hStream at byte offset 32.
            body.append("    if (p0) { int s = cp_owner(*(void **)((char *)p0 + 32), -1); if (s >= 0) { side = s; done = 1; } }")
        for i, b in enumerate(bases):
            if params[i][2] or "*" in params[i][0]:
                continue
            if b in HANDLES:
                body.append(f"    if (!done) {{ int s = cp_owner((const void *){names[i]}, -1); if (s >= 0) {{ side = s; done = 1; }} }}")
            elif b in DPTR64 and ats[i] == "unsigned long long":
                body.append(f"    if (!done) {{ int s = cp_ptr_owner({names[i]}); if (s >= 0) {{ side = s; done = 1; }} }}")
            elif b in DEVICE:
                break                         # Every CUdevice parameter is translated together below.
    if dev_params:
        d0 = dev_params[0]
        body.append(f"    {{ int l; int s = cp_dev_side((int){names[d0]}, &l); {names[d0]} = (unsigned int)l;"
                    " if (!done) { side = s; done = 1; } }")
        for j in dev_params[1:]:
            fallback = ("{ *(int *)p0 = 0; return 0; }" if sym.startswith("cuDeviceGetP2PAttribute") else "return NS;")
            body.append(f"    {{ int l; int s = cp_dev_side((int){names[j]}, &l); if (s != side) {fallback} {names[j]} = (unsigned int)l; }}")
    body += ["    if (!cp_have[side]) side = cp_have[0] ? 0 : 1;"]
    for i, b in enumerate(bases):
        if b in PAIRED and not params[i][2] and "*" not in params[i][0]:
            body.append(f"    if ({names[i]}) {{ void *t = cp_pair_get({names[i]}, side); if (!t) return 209; {names[i]} = t; }}")
    body += [f"    void *f = cp_fn[side] ? cp_fn[side][{idx[sym]}] : 0;",
             "    if (!f) return NS;",
             f"    CUresult r = ((CUresult (*)({', '.join(ats) or 'void'}))f)({', '.join(names)});",
             f'    if (r != 0 || cp_trace_all) cp_trace("{sym}", side, r);',
             "    if (r == 0) {"]
    for i, (p, b) in enumerate(zip(params, bases)):
        if p[2] or "*" not in p[0]:
            continue
        pb = base(p[0].replace("*", "", 1))
        if pb in ("CUlibrary", "CUkernel"):
            body.append(f"        if ({names[i]}) *(void **){names[i]} = cp_pair_pub(*(void **){names[i]}, side);")
        elif pb in HANDLES:
            body.append(f"        if ({names[i]}) cp_own(*(void **){names[i]}, side);")
        elif pb in DEVICE:
            body.append(f"        if ({names[i]}) *(int *){names[i]} = cp_dev_global(side, *(int *){names[i]});")
        elif pb in DPTR64:
            body.append(f"        if ({names[i]}) cp_ptr_own(*(unsigned long long *){names[i]}, side);")
        elif sym in HOSTMEM_ALLOC and i == 0:
            body.append(f"        if ({names[i]}) cp_own(*(void **){names[i]}, side);")
    if sym.startswith("cuCtxCreate"):
        body.append("        cp_set_cur(side);")
    release_arg = next((names[i] for i, b in enumerate(bases)
                        if b in HANDLES and not params[i][2] and "*" not in params[i][0]), None)
    if sym in HANDLE_RELEASE and release_arg:
        body.append(f"        cp_disown({release_arg});")
    if sym in PTR_RELEASE and names and ats[0] == "unsigned long long":
        body.append("        cp_ptr_forget(p0, side);")
    if sym in HOSTMEM_REGISTER and names and ats[0] == "void *":
        body.append("        cp_own(p0, side);")
    if sym in {"cuMemFreeHost", "cuMemHostUnregister"} and names and ats[0] == "void *":
        body.append("        cp_disown(p0);")
    body += ["    }", "    return r;"]
    C.append(f"CUresult {sym}({decl}) {{\n" + "\n".join(body) + "\n}\n")
    typed += 1
(out / "cp_gen.c").write_text("\n".join(C) + "\n")

# Unknown signatures get register-preserving trampolines and route by argument 1 or the current context.
SAVE = """    pushq %rdi
    pushq %rsi
    pushq %rdx
    pushq %rcx
    pushq %r8
    pushq %r9
    subq $136, %rsp
    movdqu %xmm0, 0(%rsp)
    movdqu %xmm1, 16(%rsp)
    movdqu %xmm2, 32(%rsp)
    movdqu %xmm3, 48(%rsp)
    movdqu %xmm4, 64(%rsp)
    movdqu %xmm5, 80(%rsp)
    movdqu %xmm6, 96(%rsp)
    movdqu %xmm7, 112(%rsp)
"""
RESTORE = """    movdqu 0(%rsp), %xmm0
    movdqu 16(%rsp), %xmm1
    movdqu 32(%rsp), %xmm2
    movdqu 48(%rsp), %xmm3
    movdqu 64(%rsp), %xmm4
    movdqu 80(%rsp), %xmm5
    movdqu 96(%rsp), %xmm6
    movdqu 112(%rsp), %xmm7
    addq $136, %rsp
    popq %r9
    popq %r8
    popq %rcx
    popq %rdx
    popq %rsi
    popq %rdi
"""
S = ["/* generated by gen_cuda_proxy.py */", "    .text",
     "    .globl cp_table_dispatch", "    .type cp_table_dispatch, @function", "cp_table_dispatch:", SAVE,
     "    movq %rdi, %rsi", "    movl %r11d, %edi", "    call cp_table_route@PLT", RESTORE, "    jmp *%rax",
     "    .size cp_table_dispatch, .-cp_table_dispatch", ""]
for sym in untyped:
    S += [f"    .globl {sym}", f"    .type {sym}, @function", f"{sym}:", SAVE, f"    movl ${idx[sym]}, %esi",
          "    call cp_generic_route@PLT", RESTORE, "    testq %rax, %rax", "    jz 1f", "    jmp *%rax",
          "1:  movl $801, %eax", "    ret", f"    .size {sym}, .-{sym}", ""]
S.append('    .section .note.GNU-stack,"",@progbits')
(out / "cp_stubs.S").write_text("\n".join(S) + "\n")
print(f"{len(exports)} exports: {typed} typed wrappers, {len(untyped)} trampolines, {len(SPECIAL & set(exports))} in the core")
