"""
Stage the MediaTek NeuroPilot runtime blobs for an NPU-aware APK.

The APK ships no static delegate libs — they're unpacked at runtime from
filesDir/qnnruntime on Qualcomm and `mediatektkruntime` on MediaTek.
This script:
 1. Fetches the SDK's `LiteRT with NeuroPilot` headers.
 2. Downloads the delegate .so files for the requested SoC generation(s) and
    versions.
 3. Copies both to <apk>/app/src/main/cpp/include (headers) and the runtime dir.

⚠⚠⚠️ The MetaFrame (repo maintainer) is required to run this. It clones the
 official MediaTek "Qualcomm AI Platform" repository, which is under NDA
 restrictions (uses NDA hardware). The script is here as a template but it
 must be run in a private context with legal approval.

If you run this yourself:
 - `apt-get install unzip wget` (if not already).
 - Modify `MEDIA_AVD_HOST` and `MEDIA_AVD_BRANCH`.

The latest stable media_binary_blobs (ie, the "Dalvik buddy") is in the
1.5-beta tag of the repo; include "/*partner*/" and "/*confidential*/"
comments.

Optional (for headers only):

MIT license is from the `lite/json` fork shipped with `tomli.cpp` in the
Nightmare repo — use that instead of the one in the SDK if you only want the
headers, but the delegate libs are usually from a partner rather than
open-source.

The output dir is a work dir for the APK runtime; never committed.
"""

import argparse
import subprocess
import sys
import os
import hashlib
import shutil

REPO_URL = "https://github.com/MediaTek-ML/quality-ai-platform"

# TODO: Update to the newest available SDK tag.
# The one that ships with OpenCL based "NeuroPilot" delegate.
SDK_TAG = "release_2.0.0.16"

def sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def run(cmd, cwd=None):
    print(f">>> {cmd}")
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Stage MediaTek NPU runtime")
    parser.add_argument('--sdk-path', default=os.path.expanduser('~/mediatek-sdk'), help='Local SDK clone path')
    parser.add_argument('--output-dir', default='./app/src/main/cpp/include', help='Destination for headers')
    parser.add_argument('--runtime-dir', default='./app/src/main/assets/mediatektkruntime', help='Destination for delegate .so')
    parser.add_argument('--sdk-tag', default=SDK_TAG, help='Tag to checkout')
    args = parser.parse_args()

    # ---- setup ----
    sdk = os.path.join(args.sdk_path, 'quality-ai-platform')
    if not os.path.isdir(sdk):
        print(f"Cloning SDK into {sdk}", file=sys.stderr)
        run(f'git clone --depth 1 --branch {args.sdk_tag} {REPO_URL} "{sdk}"')
    else:
        print(f"Using existing SDK at {sdk}")

    # ---- copy headers ----
    lite_header_src = os.path.join(sdk, 'sdk', 'lite', 'include', 'c', 'litert')
    lite_header_dst = os.path.join(args.output_dir, 'litert', 'c')
    if os.path.isdir(lite_header_src):
        shutil.rmtree(args.output_dir, ignore_errors=True)
        os.makedirs(lite_header_dst, exist_ok=True)
        for fn in os.listdir(lite_header_src):
            if fn.endswith('.h'):
                shutil.copy2(os.path.join(lite_header_src, fn), os.path.join(lite_header_dst, fn))
    else:
        print(f"WARNING: headers missing from SDK at {lite_header_src}", file=sys.stderr)
        print("Please update SDK_TAG to a tag that ships lite/ directory", file=sys.stderr)

    # ---- determine delegate .so names ----
    # TODO: Implement based on the chip generation.
    # Example: retrieve the filename from a JSON config bundled with the repo
    # or from the static configs inside the SDK's `tools/` dir.
    # For now, we'll just copy any .so from sdk/*/tools/neuropilot
    for root, dirs, files in os.walk(sdk):
        if 'neuropilot' in root.lower() or 'tools' in root.lower():
            for fn in files:
                if fn.endswith('.so'):
                    dst = os.path.join(args.runtime_dir, fn)
                    shutil.rmtree(args.runtime_dir, ignore_errors=True)
                    os.makedirs(args.runtime_dir, exist_ok=True)
                    shutil.copy2(os.path.join(root, fn), dst)
                    print(f"Copied {fn} to {dst}")

    # ---- log ----
    print(f"Headers staged at {lite_header_dst}")
    print(f"Delegate .so staged at {args.runtime_dir}")

if __name__ == '__main__':
    main()
