/**
 * YOLOv8 ONNX Model Detector
 * Handles model loading, preprocessing, and inference using ONNX Runtime Web
 * 
 * IMPORTANT: This detector supports TWO output formats:
 *  1. End2End (NMS built-in): output shape [1, N, 6] where each row is [x1, y1, x2, y2, conf, classId]
 *  2. Raw YOLO: output shape [1, 84, 8400] or [1, 8400, 84] where 84 = 4 bbox + 80 class scores
 * 
 * The code auto-detects which format is used based on output dimensions.
 */

import * as ort from 'onnxruntime-web';
import type { Detection, DetectionSettings, ModelInfo } from './types';
import { COCO_CLASSES, generateId } from './types';
import { applyNMS } from './postprocess';

// Configure ONNX Runtime Web
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
ort.env.wasm.simd = true;

export interface DetectorOptions {
  modelPath: string;
  settings: DetectionSettings;
  executionProvider?: 'webgpu' | 'wasm';
}

/** Diagnostics data exposed for the UI */
export interface DetectorDiagnostics {
  modelPath: string;
  modelLoaded: boolean;
  provider: string;
  inputName: string;
  inputShape: number[];
  outputNames: string[];
  outputShapes: number[][];
  outputFormat: 'end2end' | 'raw_yolo' | 'unknown';
  preThresholdCount: number;
  postThresholdCount: number;
  postNmsCount: number;
  lastError: string | null;
  inferenceTimeMs: number;
}

export class YOLOv8Detector {
  private session: ort.InferenceSession | null = null;
  private modelPath: string;
  private settings: DetectionSettings;
  private executionProvider: 'webgpu' | 'wasm';
  private inputName: string = 'images';
  private modelInfo: ModelInfo | null = null;
  private isInitialized: boolean = false;
  private _diagnostics: DetectorDiagnostics;
  private _firstFrame: boolean = true;

  constructor(options: DetectorOptions) {
    this.modelPath = options.modelPath;
    this.settings = options.settings;
    this.executionProvider = options.executionProvider || 'wasm';
    this._diagnostics = {
      modelPath: options.modelPath,
      modelLoaded: false,
      provider: options.executionProvider || 'wasm',
      inputName: '',
      inputShape: [],
      outputNames: [],
      outputShapes: [],
      outputFormat: 'unknown',
      preThresholdCount: 0,
      postThresholdCount: 0,
      postNmsCount: 0,
      lastError: null,
      inferenceTimeMs: 0,
    };
  }

  /** Get current diagnostics snapshot */
  getDiagnostics(): DetectorDiagnostics {
    return { ...this._diagnostics };
  }

  /**
   * Initialize the detector by loading the ONNX model
   */
  async initialize(
    onProgress?: (progress: number, status: string) => void
  ): Promise<void> {
    if (this.isInitialized) return;

    try {
      onProgress?.(0, 'Initializing ONNX Runtime...');

      console.log('=== MODEL LOADING START ===');
      console.log('  Model path:', this.modelPath);
      console.log('  Requested provider:', this.executionProvider);

      // Try WebGPU first, fallback to WASM
      const executionProviders: string[] = [];
      
      if (this.executionProvider === 'webgpu') {
        if ('gpu' in navigator) {
          executionProviders.push('webgpu');
        }
      }
      executionProviders.push('wasm');

      onProgress?.(20, 'Fetching model file...');

      // Create inference session
      this.session = await ort.InferenceSession.create(this.modelPath, {
        executionProviders,
        graphOptimizationLevel: 'all'
      });

      console.log('✅ Model loaded successfully');
      console.log('  Input names:', this.session.inputNames);
      console.log('  Output names:', this.session.outputNames);
      console.log('  Providers requested:', executionProviders);

      onProgress?.(80, 'Model loaded, configuring...');

      // Get input name from model
      const inputNames = this.session.inputNames;
      if (inputNames.length > 0) {
        this.inputName = inputNames[0];
      }

      this._diagnostics.inputName = this.inputName;
      this._diagnostics.outputNames = [...this.session.outputNames];
      this._diagnostics.modelLoaded = true;
      this._diagnostics.provider = this.executionProvider;

      // Get model info
      this.modelInfo = {
        name: 'YOLOv8n (nano)',
        version: '1.0',
        inputShape: [1, 3, this.settings.inputHeight, this.settings.inputWidth],
        numClasses: COCO_CLASSES.length,
        fileSize: 0
      };

      this._diagnostics.inputShape = [1, 3, this.settings.inputHeight, this.settings.inputWidth];

      console.log('  Model info:', this.modelInfo);
      console.log('=== MODEL LOADING COMPLETE ===');

      this.isInitialized = true;
      onProgress?.(100, 'Ready');

    } catch (error) {
      const errMsg = error instanceof Error ? error.message : 'Unknown error';
      console.error('❌ Model loading FAILED:', errMsg);
      this._diagnostics.lastError = `Model load failed: ${errMsg}`;
      this._diagnostics.modelLoaded = false;
      throw new Error(`Detector initialization failed: ${errMsg}`);
    }
  }

  isReady(): boolean {
    return this.isInitialized && this.session !== null;
  }

  getModelInfo(): ModelInfo | null {
    return this.modelInfo;
  }

  getExecutionProvider(): string {
    return this.executionProvider;
  }

  /**
   * Preprocess image data for YOLOv8 input
   * Converts ImageData to NCHW tensor with normalization to [0, 1]
   */
  preprocess(imageData: ImageData): {
    tensor: ort.Tensor;
    scaleX: number;
    scaleY: number;
    padX: number;
    padY: number;
  } {
    const { width, height, data } = imageData;
    const { inputWidth, inputHeight } = this.settings;

    // Calculate scale factors (letterbox padding)
    const scaleX = inputWidth / width;
    const scaleY = inputHeight / height;
    const scale = Math.min(scaleX, scaleY);
    
    const newWidth = Math.round(width * scale);
    const newHeight = Math.round(height * scale);
    
    const padX = Math.round((inputWidth - newWidth) / 2);
    const padY = Math.round((inputHeight - newHeight) / 2);

    // Create input tensor with NCHW format [1, 3, 640, 640]
    const channels = 3;
    const inputSize = inputWidth * inputHeight * channels;
    const float32Data = new Float32Array(inputSize);

    // Initialize with 0s (for letterbox padding area)
    float32Data.fill(0);

    // Process image data with letterbox resize
    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        // Map from scaled coordinates back to original
        const srcX = Math.floor(x / scale);
        const srcY = Math.floor(y / scale);
        
        if (srcX >= width || srcY >= height) continue;
        
        const srcIdx = (srcY * width + srcX) * 4;
        const dstY = y + padY;
        const dstX = x + padX;
        const dstIdx = dstY * inputWidth + dstX;

        // Normalize to [0, 1] and arrange in NCHW: R plane, G plane, B plane
        float32Data[dstIdx] = data[srcIdx] / 255;                                    // R
        float32Data[inputWidth * inputHeight + dstIdx] = data[srcIdx + 1] / 255;     // G
        float32Data[2 * inputWidth * inputHeight + dstIdx] = data[srcIdx + 2] / 255; // B
      }
    }

    const tensor = new ort.Tensor(
      'float32',
      float32Data,
      [1, channels, inputHeight, inputWidth]
    );

    return { tensor, scaleX: scale, scaleY: scale, padX, padY };
  }

  /**
   * Postprocess END2END model output: [1, N, 6]
   * Each detection is [x1, y1, x2, y2, confidence, classId]
   * NMS is already applied by the model — we only need to threshold and convert coords.
   */
  private postprocessEnd2End(
    output: ort.Tensor,
    originalWidth: number,
    originalHeight: number,
    scale: number,
    padX: number,
    padY: number
  ): Detection[] {
    const outputData = output.data as Float32Array;
    const dims = output.dims as number[];
    const numDetections = dims[1]; // e.g. 300
    const stride = dims[2];       // 6: [x1, y1, x2, y2, conf, classId]

    const { confidenceThreshold, maxDetections } = this.settings;

    const detections: Detection[] = [];
    let preThresholdCount = 0;

    for (let i = 0; i < numDetections; i++) {
      const base = i * stride;
      const x1_model = outputData[base + 0]; // x1 in model coords (0-640)
      const y1_model = outputData[base + 1]; // y1
      const x2_model = outputData[base + 2]; // x2
      const y2_model = outputData[base + 3]; // y2
      const confidence = outputData[base + 4];
      const classIdFloat = outputData[base + 5];
      const classId = Math.round(classIdFloat);

      // Skip zero-confidence padding slots (model pads to N=300)
      if (confidence <= 0) continue;

      preThresholdCount++;

      // Apply confidence threshold
      if (confidence < confidenceThreshold) continue;

      // Convert from model coordinates (with padding) back to original image
      const x1 = (x1_model - padX) / scale;
      const y1 = (y1_model - padY) / scale;
      const x2 = (x2_model - padX) / scale;
      const y2 = (y2_model - padY) / scale;

      const bboxWidth = x2 - x1;
      const bboxHeight = y2 - y1;

      // Skip invalid boxes
      if (bboxWidth <= 0 || bboxHeight <= 0) continue;

      // Clip to image bounds
      const clippedX = Math.max(0, x1);
      const clippedY = Math.max(0, y1);
      const clippedW = Math.min(bboxWidth, originalWidth - clippedX);
      const clippedH = Math.min(bboxHeight, originalHeight - clippedY);

      if (clippedW <= 0 || clippedH <= 0) continue;

      const label = (classId >= 0 && classId < COCO_CLASSES.length)
        ? COCO_CLASSES[classId]
        : `class_${classId}`;

      detections.push({
        id: generateId(),
        bbox: {
          x: clippedX,
          y: clippedY,
          width: clippedW,
          height: clippedH
        },
        label,
        classId,
        confidence,
        timestamp: Date.now()
      });
    }

    this._diagnostics.preThresholdCount = preThresholdCount;
    this._diagnostics.postThresholdCount = detections.length;
    // NMS is already done by the model for end2end format
    this._diagnostics.postNmsCount = detections.length;

    return detections.slice(0, maxDetections);
  }

  /**
   * Postprocess RAW YOLO output: [1, 84, 8400] or [1, 8400, 84]
   * 84 = 4 bbox (cx, cy, w, h) + 80 class scores
   * Requires manual NMS.
   */
  private postprocessRawYolo(
    output: ort.Tensor,
    originalWidth: number,
    originalHeight: number,
    scale: number,
    padX: number,
    padY: number
  ): Detection[] {
    const { confidenceThreshold, iouThreshold, maxDetections } = this.settings;
    const outputData = output.data as Float32Array;
    const dims = output.dims as number[];

    let numAnchors: number;
    let numFeatures: number;
    let layout: 'channels_first' | 'anchors_first';

    if (dims[1] < dims[2]) {
      numFeatures = dims[1];
      numAnchors = dims[2];
      layout = 'channels_first';
    } else {
      numAnchors = dims[1];
      numFeatures = dims[2];
      layout = 'anchors_first';
    }

    const numClasses = numFeatures - 4;
    if (numClasses <= 0) {
      throw new Error(`Invalid numClasses: ${numClasses} (numFeatures=${numFeatures})`);
    }

    const detections: Detection[] = [];
    let preThresholdCount = 0;

    for (let i = 0; i < numAnchors; i++) {
      let cx: number, cy: number, w: number, h: number;

      if (layout === 'channels_first') {
        cx = outputData[0 * numAnchors + i];
        cy = outputData[1 * numAnchors + i];
        w = outputData[2 * numAnchors + i];
        h = outputData[3 * numAnchors + i];
      } else {
        const base = i * numFeatures;
        cx = outputData[base + 0];
        cy = outputData[base + 1];
        w = outputData[base + 2];
        h = outputData[base + 3];
      }

      let maxScore = 0;
      let maxClassId = 0;

      for (let c = 0; c < numClasses; c++) {
        let score: number;
        if (layout === 'channels_first') {
          score = outputData[(4 + c) * numAnchors + i];
        } else {
          score = outputData[i * numFeatures + 4 + c];
        }
        if (score > maxScore) {
          maxScore = score;
          maxClassId = c;
        }
      }

      preThresholdCount++;

      if (!Number.isFinite(maxScore) || maxScore < confidenceThreshold) continue;

      const x = (cx - w / 2 - padX) / scale;
      const y = (cy - h / 2 - padY) / scale;
      const scaledW = w / scale;
      const scaledH = h / scale;

      if (scaledW <= 0 || scaledH <= 0) continue;

      const clippedX = Math.max(0, x);
      const clippedY = Math.max(0, y);
      const clippedW = Math.min(scaledW, originalWidth - clippedX);
      const clippedH = Math.min(scaledH, originalHeight - clippedY);

      if (clippedW <= 0 || clippedH <= 0) continue;

      detections.push({
        id: generateId(),
        bbox: { x: clippedX, y: clippedY, width: clippedW, height: clippedH },
        label: COCO_CLASSES[maxClassId] || `class_${maxClassId}`,
        classId: maxClassId,
        confidence: maxScore,
        timestamp: Date.now()
      });
    }

    this._diagnostics.preThresholdCount = preThresholdCount;
    this._diagnostics.postThresholdCount = detections.length;

    const nmsDetections = applyNMS(detections, iouThreshold);
    this._diagnostics.postNmsCount = nmsDetections.length;

    return nmsDetections.slice(0, maxDetections);
  }

  /**
   * Full detection pipeline: preprocess -> inference -> postprocess
   */
  async detect(imageData: ImageData): Promise<{
    detections: Detection[];
    preprocessTime: number;
    inferenceTime: number;
    postprocessTime: number;
  }> {
    if (!this.isReady()) {
      throw new Error('Detector not initialized');
    }

    this._diagnostics.lastError = null;

    try {
      // Preprocess
      const preprocessStart = performance.now();
      const { tensor, scaleX, padX, padY } = this.preprocess(imageData);
      const preprocessTime = performance.now() - preprocessStart;

      // Inference
      const inferenceStart = performance.now();
      const feeds: Record<string, ort.Tensor> = {};
      feeds[this.inputName] = tensor;
      const results = await this.session!.run(feeds);
      const inferenceTime = performance.now() - inferenceStart;

      // Dispose input tensor immediately
      tensor.dispose();

      this._diagnostics.inferenceTimeMs = inferenceTime;

      // Get the first output tensor
      const outputNames = this.session!.outputNames;
      const output = results[outputNames[0]];
      const dims = output.dims as number[];

      // Log on first frame only — forensic output for verification
      if (this._firstFrame) {
        console.log('=== FIRST FRAME INFERENCE ===');
        console.log('  Input tensor name:', this.inputName);
        console.log('  Input tensor shape:', [1, 3, this.settings.inputHeight, this.settings.inputWidth]);
        console.log('  Output names:', outputNames);
        for (const name of outputNames) {
          const o = results[name];
          console.log(`  Output "${name}":`, {
            dims: o?.dims,
            type: o?.type,
            dataLength: o?.data?.length,
          });
        }

        // Log first 10 raw values for forensic verification
        const raw = output.data as Float32Array;
        console.log('  First 30 raw output values:', Array.from(raw.slice(0, 30)));
        this._firstFrame = false;
      }

      // Store output shapes for diagnostics
      this._diagnostics.outputShapes = outputNames.map(name => {
        const o = results[name];
        return o?.dims ? [...(o.dims as number[])] : [];
      });

      // Postprocess
      const postprocessStart = performance.now();
      let detections: Detection[];

      // Auto-detect output format based on dimensions
      if (dims.length === 3 && dims[2] === 6) {
        // End2End format: [1, N, 6] → [x1, y1, x2, y2, conf, classId]
        this._diagnostics.outputFormat = 'end2end';
        detections = this.postprocessEnd2End(
          output, imageData.width, imageData.height, scaleX, padX, padY
        );
      } else if (dims.length === 3 && (dims[1] === 84 || dims[2] === 84)) {
        // Raw YOLO format: [1, 84, 8400] or [1, 8400, 84]
        this._diagnostics.outputFormat = 'raw_yolo';
        detections = this.postprocessRawYolo(
          output, imageData.width, imageData.height, scaleX, padX, padY
        );
      } else {
        this._diagnostics.outputFormat = 'unknown';
        const errMsg = `Unsupported output shape: [${dims.join(', ')}]. Expected [1, N, 6] (end2end) or [1, 84, 8400] (raw YOLO).`;
        this._diagnostics.lastError = errMsg;
        throw new Error(errMsg);
      }

      const postprocessTime = performance.now() - postprocessStart;

      return { detections, preprocessTime, inferenceTime, postprocessTime };

    } catch (error) {
      const errMsg = error instanceof Error ? error.message : 'Unknown inference error';
      this._diagnostics.lastError = errMsg;
      throw error; // Re-throw — do NOT return fake detections
    }
  }

  updateSettings(newSettings: Partial<DetectionSettings>): void {
    this.settings = { ...this.settings, ...newSettings };
  }

  getSettings(): DetectionSettings {
    return { ...this.settings };
  }

  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
      this.isInitialized = false;
      this._diagnostics.modelLoaded = false;
    }
  }
}
