/* Look Pilot: local-only, cancellable U²-Net inference. No image leaves this worker. */
/* Runtime/model licences and hashes are in this directory's README.md. */
"use strict";
self.onmessage = async (event) => {
  let session;
  try {
    self.postMessage({ type: "progress", stage: "loading" });
    importScripts("./ort.wasm.min.js");
    ort.env.wasm.wasmPaths = new URL("./", self.location.href).href;
    ort.env.wasm.numThreads = 1; // Works without SharedArrayBuffer / isolation headers.
    ort.env.wasm.proxy = false; // Already in a dedicated, terminable worker.
    const response = await fetch(new URL("./u2netp.onnx", self.location.href));
    if (!response.ok) throw new Error("The local image tools could not be loaded. Please retry.");
    session = await ort.InferenceSession.create(await response.arrayBuffer(), { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    self.postMessage({ type: "progress", stage: "segmenting" });
    const tensor = new ort.Tensor("float32", event.data.input, [1, 3, 320, 320]);
    const outputs = await session.run({ [session.inputNames[0]]: tensor });
    const output = outputs[session.outputNames[0]];
    const mask = new Float32Array(output.data);
    tensor.dispose();
    for (const value of Object.values(outputs)) value.dispose();
    self.postMessage({ type: "result", mask }, [mask.buffer]);
  } catch (error) {
    self.postMessage({ type: "error", message: "The cutout could not run on this device. Keep the original, or retry after reloading." });
  } finally {
    if (session) await session.release();
  }
};
