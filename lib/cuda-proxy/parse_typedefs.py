#!/usr/bin/env python3
"""Pull every versioned CUDA driver signature out of cudaTypedefs.h for the wrapper generator."""
import json, re, sys
src = open(sys.argv[1]).read()
src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
out = []
for m in re.finditer(r"typedef\s+(\w+)\s*\(\s*CUDAAPI\s*\*\s*PFN_(\w+?)_v(\d+)(_ptsz|_ptds)?\s*\)\s*\((.*?)\)\s*;", src, re.S):
    ret, name, ver, pt, params = m.group(1), m.group(2), int(m.group(3)), m.group(4), " ".join(m.group(5).split())
    plist = []
    if params and params != "void":
        depth, cur = 0, ""
        for ch in params:                       # Only split top-level commas; callback types contain their own.
            if ch == "(": depth += 1
            if ch == ")": depth -= 1
            if ch == "," and depth == 0: plist.append(cur.strip()); cur = ""
            else: cur += ch
        if cur.strip(): plist.append(cur.strip())
    parsed = []
    for p in plist:
        fp = re.match(r"(.*\(\s*CUDA_CB\s*\*\s*|.*\(\s*\*\s*)(\w+)(\s*\).*)", p)
        if fp:                                  # Keep callback declarations intact.
            parsed.append([fp.group(1) + fp.group(3), fp.group(2), True]); continue
        mm = re.match(r"(.*?)(\w+)(\s*\[\s*\w*\s*\])?$", p)
        typ, arg = (mm.group(1).strip(), mm.group(2)) if mm else (p, "")
        if mm and mm.group(3): typ += " *"
        if not typ: typ, arg = arg, ""
        parsed.append([typ, arg, False])
    out.append({"name": name, "ver": ver, "pt": 1 if pt else 0, "ret": ret, "params": parsed})
json.dump(out, sys.stdout, indent=0)
