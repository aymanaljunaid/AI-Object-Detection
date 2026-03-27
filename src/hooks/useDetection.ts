/**
 * Object Detection Hook
 * Manages detection loop with FPS tracking and performance monitoring
 * Uses real YOLOv8 model inference ONLY — no demo/mock/fake mode
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type { Detection, DetectionSettings, PerformanceMetrics, DetectionEvent } from '@/lib/detection/types';
import { DEFAULT_SETTINGS, generateId } from '@/lib/detection/types';
import { YOLOv8Detector, type DetectorDiagnostics } from '@/lib/detection/detector';

export interface UseDetectionOptions {
  modelPath: string;
  settings?: Partial<DetectionSettings>;
  onDetection?: (detections: Detection[]) => void;
  onError?: (error: Error) => void;
  autoLog?: boolean;
}

export interface UseDetectionReturn {
  // State
  isModelLoading: boolean;
  isDetecting: boolean;
  modelLoadProgress: number;
  modelLoadStatus: string;
  modelError: string | null;
  detections: Detection[];
  metrics: PerformanceMetrics;
  settings: DetectionSettings;
  executionProvider: string;
  eventLog: DetectionEvent[];
  diagnostics: DetectorDiagnostics | null;
  // BUG 1 FIX: expose isModelReady so callers can gate startDetection()
  isModelReady: boolean;

  // Actions
  loadModel: () => Promise<void>;
  startDetection: () => void;
  stopDetection: () => void;
  updateSettings: (newSettings: Partial<DetectionSettings>) => void;
  clearEventLog: () => void;
  detectFrame: (imageData: ImageData) => Promise<Detection[]>;
}

const EMPTY_DIAGNOSTICS: DetectorDiagnostics = {
  modelPath: '',
  modelLoaded: false,
  provider: 'none',
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

export function useDetection(options: UseDetectionOptions): UseDetectionReturn {
  const {
    modelPath,
    settings: initialSettings,
    onDetection,
    onError,
    autoLog = true
  } = options;

  // State
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [modelLoadProgress, setModelLoadProgress] = useState(0);
  const [modelLoadStatus, setModelLoadStatus] = useState('');
  const [modelError, setModelError] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [executionProvider, setExecutionProvider] = useState<string>('wasm');
  const [eventLog, setEventLog] = useState<DetectionEvent[]>([]);
  const [diagnostics, setDiagnostics] = useState<DetectorDiagnostics | null>(null);
  // BUG 1 FIX: track whether the model is actually ready
  const [isModelReady, setIsModelReady] = useState(false);

  const [settings, setSettings] = useState<DetectionSettings>({
    ...DEFAULT_SETTINGS,
    ...initialSettings
  });

  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 0,
    inferenceTime: 0,
    preprocessTime: 0,
    postprocessTime: 0,
    frameCount: 0
  });

  // Refs
  const detectorRef = useRef<YOLOv8Detector | null>(null);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(performance.now());
  const fpsFrameCountRef = useRef(0);
  const animationFrameRef = useRef<number | null>(null);
  const sessionIdRef = useRef<string>(generateId());
  const isInferenceRunningRef = useRef(false);
  const latestMetricsRef = useRef({ inferenceTime: 0, preprocessTime: 0, postprocessTime: 0 });

  // Load the model
  const loadModel = useCallback(async () => {
    if (isModelLoading || detectorRef.current?.isReady()) return;

    setIsModelLoading(true);
    setIsModelReady(false);
    setModelError(null);
    setModelLoadProgress(0);
    setDiagnostics({ ...EMPTY_DIAGNOSTICS, modelPath });

    try {
      // BUG 7 FIX: properly check WebGPU availability via requestAdapter()
      // instead of just checking if 'gpu' exists in navigator.
      let provider: 'webgpu' | 'wasm' = 'wasm';
      if ('gpu' in navigator) {
        try {
          const adapter = await (navigator as unknown as { gpu: { requestAdapter: () => Promise<unknown> } }).gpu.requestAdapter();
          if (adapter) {
            provider = 'webgpu';
          }
        } catch {
          // WebGPU not functional, fall back to wasm
          provider = 'wasm';
        }
      }

      console.log('🔧 Loading model:', modelPath);
      console.log('🔧 Execution provider:', provider);

      detectorRef.current = new YOLOv8Detector({
        modelPath,
        settings,
        executionProvider: provider
      });

      await detectorRef.current.initialize((progress, status) => {
        setModelLoadProgress(progress);
        setModelLoadStatus(status);
      });

      setExecutionProvider(detectorRef.current.getExecutionProvider());
      setDiagnostics(detectorRef.current.getDiagnostics());
      setIsModelReady(true);
      console.log('✅ Model loaded successfully');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load model';
      console.error('❌ Model loading failed:', errorMessage);
      setModelError(errorMessage);
      setIsModelReady(false);
      if (detectorRef.current) {
        setDiagnostics(detectorRef.current.getDiagnostics());
      }
      onError?.(error instanceof Error ? error : new Error(errorMessage));
    } finally {
      setIsModelLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isModelLoading, modelPath, onError]);

  // Detect on a single frame — REAL INFERENCE ONLY, NO FALLBACKS
  const detectFrame = useCallback(async (imageData: ImageData): Promise<Detection[]> => {
    // Guard: skip if previous inference is still running
    if (isInferenceRunningRef.current) return [];

    if (!detectorRef.current?.isReady()) {
      console.error('❌ Detector not ready');
      onError?.(new Error('Detector not initialized. Please load the model first.'));
      return [];
    }

    isInferenceRunningRef.current = true;

    try {
      const result = await detectorRef.current.detect(imageData);

      // Update diagnostics from detector
      setDiagnostics(detectorRef.current.getDiagnostics());

      // Track frame counts for FPS calculation
      frameCountRef.current++;
      fpsFrameCountRef.current++;

      // Store latest timing metrics in ref (no re-render)
      latestMetricsRef.current = {
        inferenceTime: result.inferenceTime,
        preprocessTime: result.preprocessTime,
        postprocessTime: result.postprocessTime,
      };

      // Only update React state once per second (for FPS display)
      const now = performance.now();
      const elapsed = now - lastFpsUpdateRef.current;

      if (elapsed >= 1000) {
        const fps = (fpsFrameCountRef.current * 1000) / elapsed;

        setMetrics(prev => ({
          fps,
          inferenceTime: latestMetricsRef.current.inferenceTime,
          preprocessTime: latestMetricsRef.current.preprocessTime,
          postprocessTime: latestMetricsRef.current.postprocessTime,
          frameCount: prev.frameCount + fpsFrameCountRef.current
        }));

        fpsFrameCountRef.current = 0;
        lastFpsUpdateRef.current = now;
      }

      // Log detections if enabled — but deduplicate by label per frame
      if (autoLog && result.detections.length > 0) {
        // Only log unique class detections per frame to avoid spam
        const seenLabels = new Set<string>();
        const events: DetectionEvent[] = [];

        for (const d of result.detections) {
          if (!seenLabels.has(d.label)) {
            seenLabels.add(d.label);
            events.push({
              id: generateId(),
              label: d.label,
              classId: d.classId,
              confidence: d.confidence,
              bbox: d.bbox,
              timestamp: d.timestamp,
              sessionId: sessionIdRef.current
            });
          }
        }

        if (events.length > 0) {
          setEventLog(prev => [...prev.slice(-499), ...events]);
        }
      }

      setDetections(result.detections);
      onDetection?.(result.detections);
      return result.detections;

    } catch (error) {
      console.error('❌ Detection error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Detection failed';
      if (detectorRef.current) {
        setDiagnostics(detectorRef.current.getDiagnostics());
      }
      onError?.(new Error(errorMessage));
      // Return empty — NEVER return fake detections
      setDetections([]);
      return [];
    } finally {
      isInferenceRunningRef.current = false;
    }
  }, [autoLog, onDetection, onError]);

  // Start continuous detection loop
  const startDetection = useCallback(() => {
    if (isDetecting) return;
    setIsDetecting(true);
    sessionIdRef.current = generateId();
  }, [isDetecting]);

  // Stop detection loop
  const stopDetection = useCallback(() => {
    setIsDetecting(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    setDetections([]);
  }, []);

  // Update settings
  const updateSettings = useCallback((newSettings: Partial<DetectionSettings>) => {
    setSettings(prev => {
      const updated = { ...prev, ...newSettings };
      detectorRef.current?.updateSettings(updated);
      return updated;
    });
  }, []);

  // Clear event log
  const clearEventLog = useCallback(() => {
    setEventLog([]);
  }, []);

  // BUG 8 FIX: dispose() is async but React cleanup can't await.
  // We store the promise in a local variable and let it resolve on its own,
  // preventing the linter warning and ensuring cleanup still runs.
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      void detectorRef.current?.dispose();
    };
  }, []);

  return {
    isModelLoading,
    isDetecting,
    modelLoadProgress,
    modelLoadStatus,
    modelError,
    detections,
    metrics,
    settings,
    executionProvider,
    eventLog,
    diagnostics,
    isModelReady,
    loadModel,
    startDetection,
    stopDetection,
    updateSettings,
    clearEventLog,
    detectFrame
  };
}
