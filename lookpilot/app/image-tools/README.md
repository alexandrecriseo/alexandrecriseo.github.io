# Local image tools

All assets are served from Look Pilot's own origin. The worker performs inference on the device and never uploads an image. Assets load only after **Remove background**.

- `u2netp.onnx`: U²-Net small from the [rembg maintainer release](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx). Upstream model/code: [Xuebin Qin et al., U²-Net](https://github.com/xuebinqin/U-2-Net), Apache-2.0; licence retained as `U2NET-LICENSE.txt`.
- SHA-256: `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`; MD5 matches the rembg maintainer's session loader: `8e83ca70e441ab06c318d82300c84806`.
- `ort.wasm.min.js`, `ort-wasm-simd-threaded.mjs` and `ort-wasm-simd-threaded.wasm`: ONNX Runtime Web **1.30.0**, Microsoft, MIT; `ONNXRUNTIME-LICENSE.txt` is the official version's licence. Copied unchanged from the pinned npm package by `node scripts/image-tools.cjs` during explicit builds. The runtime works in a dedicated worker using single-thread WASM, without SharedArrayBuffer or WebGPU.
- `background-removal.worker.js`: application integration, not copied rembg or IMG.LY code.

The original image is not a model output. Only a mask is inferred, multiplied into its existing alpha, then uniformly resized onto a 720 × 720 transparent canvas. Natural proportions and source RGB are preserved, subject to ordinary image resampling. This is foreground extraction, not clothing recognition or removal of a wearer/mannequin.

See [background-removal evaluation](../../docs/background-removal.md) for limits and measured evidence.
