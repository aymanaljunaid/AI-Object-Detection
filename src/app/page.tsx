'use client';

/**
 * AI Real-Time Object Detection
 * Main page with camera, detection, controls, and diagnostics
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  AlertCircle,
  Camera,
  Download,
  Loader2,
  Play,
  Square,
  Trash2,
  Monitor,
  Info,
  Activity,
  Eye,
  Cpu,
  Layers
} from 'lucide-react';
import { useCamera } from '@/hooks/useCamera';
import { useDetection } from '@/hooks/useDetection';
import { DetectionCanvas } from '@/components/DetectionCanvas';
import { ThemeToggle } from '@/components/ThemeToggle';
import { COCO_CLASSES, getClassColor } from '@/lib/detection/types';
import { exportEvents, getSummary, clearEvents, saveEvents } from '@/lib/detection/eventLogger';

const MODEL_PATH = '/models/yolov8n.onnx';

export default function Home() {
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const detectionLoopRef = useRef<number | null>(null);

  const {
    videoRef,
    isStreaming,
    isLoading: cameraLoading,
    error: cameraError,
    capabilities,
    startCamera,
    stopCamera,
    captureFrame,
    getVideoSize
  } = useCamera({ width: 640, height: 480, frameRate: 30 });

  const {
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
    loadModel,
    startDetection,
    stopDetection,
    updateSettings,
    clearEventLog,
    detectFrame
  } = useDetection({
    modelPath: MODEL_PATH,
    settings: {
      confidenceThreshold: 0.5,
      iouThreshold: 0.45,
      maxDetections: 100
    },
    autoLog: true
  });

  const [showLabels, setShowLabels] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);
  const [showFps, setShowFps] = useState(true);
  const [videoSize, setVideoSize] = useState({ width: 640, height: 480 });
  const [stats, setStats] = useState<ReturnType<typeof getSummary> | null>(null);

  // Update video size when streaming starts
  useEffect(() => {
    if (isStreaming) {
      const updateSize = () => {
        const size = getVideoSize();
        setVideoSize(size);
      };
      updateSize();
      const interval = setInterval(updateSize, 500);
      return () => clearInterval(interval);
    }
  }, [isStreaming, getVideoSize]);

  // Refs to avoid stale closures in detection loop
  const isDetectingRef = useRef(isDetecting);
  const isStreamingRef = useRef(isStreaming);
  const captureFrameRef = useRef(captureFrame);
  const detectFrameRef = useRef(detectFrame);

  useEffect(() => {
    isDetectingRef.current = isDetecting;
    isStreamingRef.current = isStreaming;
    captureFrameRef.current = captureFrame;
    detectFrameRef.current = detectFrame;
  }, [isDetecting, isStreaming, captureFrame, detectFrame]);

  const runDetectionLoop = useCallback(() => {
    const loop = async () => {
      if (!isDetectingRef.current || !isStreamingRef.current) return;
      const frame = captureFrameRef.current();
      if (frame) {
        await detectFrameRef.current(frame);
      }
      detectionLoopRef.current = requestAnimationFrame(loop);
    };
    loop();
  }, []);

  useEffect(() => {
    if (isDetecting && isStreaming) {
      runDetectionLoop();
    }
    return () => {
      if (detectionLoopRef.current) {
        cancelAnimationFrame(detectionLoopRef.current);
        detectionLoopRef.current = null;
      }
    };
  }, [isDetecting, isStreaming, runDetectionLoop]);

  useEffect(() => {
    const updateStats = () => setStats(getSummary());
    updateStats();
    const interval = setInterval(updateStats, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (eventLog.length > 0) saveEvents(eventLog);
  }, [eventLog]);

  const handleStart = useCallback(async () => {
    if (!isModelLoading && !isDetecting) await loadModel();
    if (!isStreaming) await startCamera();
    startDetection();
  }, [isModelLoading, isDetecting, isStreaming, loadModel, startCamera, startDetection]);

  const handleStop = useCallback(() => {
    stopDetection();
    stopCamera();
  }, [stopDetection, stopCamera]);

  const handleExport = useCallback((format: 'json' | 'csv') => {
    const data = exportEvents(format);
    const blob = new Blob([data], { type: format === 'json' ? 'application/json' : 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `detection_events.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  const handleClearAll = useCallback(() => {
    clearEventLog();
    clearEvents();
    setStats(null);
  }, [clearEventLog]);

  const error = cameraError || modelError;

  return (
    <div className="min-h-screen bg-background transition-colors duration-300">
      {/* ─── Top Bar ─── */}
      <header className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur-lg">
        <div className="container mx-auto px-4 max-w-7xl h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 relative shrink-0">
              {/* White logo for dark mode */}
              <img
                src="/logos/object-detection-black.png"
                alt="Logo"
                className="h-9 w-9 hidden dark:block"
              />
              {/* Black logo for light mode */}
              <img
                src="/logos/object-detection-white.png"
                alt="Logo"
                className="h-9 w-9 block dark:hidden"
              />
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight leading-tight">
                AI Object Detection
              </h1>
              <p className="text-xs text-muted-foreground leading-tight">
                YOLOv8n · {executionProvider === 'webgpu' ? 'WebGPU' : 'WASM'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={executionProvider === 'webgpu' ? 'default' : 'secondary'} className="text-xs">
              <Cpu className="h-3 w-3 mr-1" />
              {executionProvider === 'webgpu' ? 'WebGPU' : 'WASM'}
            </Badge>
            <Badge variant="outline" className="text-xs">
              <Layers className="h-3 w-3 mr-1" />
              {COCO_CLASSES.length} classes
            </Badge>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <div className="container mx-auto p-4 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* ─── Main Column ─── */}
          <div className="lg:col-span-2 space-y-4">
            {/* Camera Feed */}
            <Card className="overflow-hidden">
              <CardHeader className="pb-2 px-4 pt-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Camera className="h-4 w-4 text-muted-foreground" />
                    Camera Feed
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    {isStreaming && (
                      <Badge className="bg-emerald-500/90 text-white text-xs animate-pulse-glow border-0">
                        <span className="mr-1.5 animate-pulse">●</span> Live
                      </Badge>
                    )}
                    {isDetecting && (
                      <Badge variant="outline" className="text-xs">
                        {detections.length} object{detections.length !== 1 ? 's' : ''}
                      </Badge>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <div
                  ref={videoContainerRef}
                  className="relative bg-black rounded-xl overflow-hidden aspect-video"
                >
                  <video
                    ref={videoRef}
                    className="absolute inset-0 w-full h-full object-contain"
                    autoPlay
                    playsInline
                    muted
                  />

                  {isDetecting && (
                    <DetectionCanvas
                      detections={detections}
                      width={videoSize.width}
                      height={videoSize.height}
                      showLabels={showLabels}
                      showConfidence={showConfidence}
                      showFps={showFps}
                      metrics={metrics}
                      className="w-full h-full object-contain"
                    />
                  )}

                  {isModelLoading && (
                    <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center">
                      <Loader2 className="h-10 w-10 animate-spin text-primary mb-3" />
                      <p className="text-white text-sm font-medium">{modelLoadStatus}</p>
                      <div className="w-48 h-1.5 bg-white/10 rounded-full mt-3 overflow-hidden">
                        <div
                          className="h-full bg-primary rounded-full transition-all duration-300"
                          style={{ width: `${modelLoadProgress}%` }}
                        />
                      </div>
                      <p className="text-white/50 text-xs mt-2">{modelLoadProgress}%</p>
                    </div>
                  )}

                  {error && (
                    <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center p-6">
                      <AlertCircle className="h-10 w-10 text-destructive mb-3" />
                      <p className="text-white text-sm font-medium text-center max-w-md">{error}</p>
                      <Button variant="outline" size="sm" className="mt-4" onClick={() => window.location.reload()}>
                        Retry
                      </Button>
                    </div>
                  )}

                  {!isStreaming && !isModelLoading && !error && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <div className="h-16 w-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-3">
                        <Camera className="h-8 w-8 text-white/30" />
                      </div>
                      <p className="text-white/40 text-sm">Camera not started</p>
                      <p className="text-white/20 text-xs mt-1">Click "Start" to begin detection</p>
                    </div>
                  )}
                </div>

                {/* Controls */}
                <div className="flex items-center justify-center gap-3 mt-4">
                  {!isDetecting ? (
                    <Button
                      size="lg"
                      onClick={handleStart}
                      disabled={isModelLoading || cameraLoading}
                      className="min-w-[140px] rounded-full"
                    >
                      {isModelLoading || cameraLoading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Loading…
                        </>
                      ) : (
                        <>
                          <Play className="mr-2 h-4 w-4" />
                          Start
                        </>
                      )}
                    </Button>
                  ) : (
                    <Button
                      size="lg"
                      variant="destructive"
                      onClick={handleStop}
                      className="min-w-[140px] rounded-full"
                    >
                      <Square className="mr-2 h-4 w-4" />
                      Stop
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Settings */}
            <Card>
              <CardHeader className="pb-2 px-4 pt-4">
                <CardTitle className="text-base flex items-center gap-2">
                  <Activity className="h-4 w-4 text-muted-foreground" />
                  Detection Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Confidence Threshold */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium">Confidence</label>
                      <span className="text-sm font-mono text-primary">
                        {(settings.confidenceThreshold * 100).toFixed(0)}%
                      </span>
                    </div>
                    <Slider
                      value={[settings.confidenceThreshold * 100]}
                      onValueChange={([value]) => updateSettings({ confidenceThreshold: value / 100 })}
                      min={10}
                      max={90}
                      step={5}
                    />
                  </div>

                  {/* IoU Threshold */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium">IoU (NMS)</label>
                      <span className="text-sm font-mono text-primary">
                        {(settings.iouThreshold * 100).toFixed(0)}%
                      </span>
                    </div>
                    <Slider
                      value={[settings.iouThreshold * 100]}
                      onValueChange={([value]) => updateSettings({ iouThreshold: value / 100 })}
                      min={20}
                      max={80}
                      step={5}
                    />
                  </div>

                  {/* Display Options */}
                  <div className="space-y-2.5">
                    <label className="text-sm font-medium">Display</label>
                    {[
                      { label: 'Labels', checked: showLabels, onChange: setShowLabels },
                      { label: 'Confidence', checked: showConfidence, onChange: setShowConfidence },
                      { label: 'FPS overlay', checked: showFps, onChange: setShowFps },
                    ].map(({ label, checked, onChange }) => (
                      <div key={label} className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">{label}</span>
                        <Switch checked={checked} onCheckedChange={onChange} />
                      </div>
                    ))}
                  </div>

                  {/* Data Management */}
                  <div className="space-y-2.5">
                    <label className="text-sm font-medium">Data</label>
                    <div className="flex gap-2 flex-wrap">
                      <Button variant="outline" size="sm" onClick={() => handleExport('json')} disabled={!stats?.totalEvents}>
                        <Download className="mr-1.5 h-3.5 w-3.5" />
                        JSON
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => handleExport('csv')} disabled={!stats?.totalEvents}>
                        <Download className="mr-1.5 h-3.5 w-3.5" />
                        CSV
                      </Button>
                      <Button variant="destructive" size="sm" onClick={handleClearAll} disabled={!stats?.totalEvents}>
                        <Trash2 className="mr-1.5 h-3.5 w-3.5" />
                        Clear
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Model Info */}
            <Card>
              <CardHeader className="pb-2 px-4 pt-4">
                <CardTitle className="text-base flex items-center gap-2">
                  <Info className="h-4 w-4 text-muted-foreground" />
                  Model
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {[
                    { label: 'Architecture', value: 'YOLOv8n (nano)' },
                    { label: 'Classes', value: `${COCO_CLASSES.length} (COCO)` },
                    { label: 'Input', value: '640 × 640' },
                    { label: 'Provider', value: executionProvider === 'webgpu' ? 'WebGPU' : 'WASM' },
                  ].map(({ label, value }) => (
                    <div key={label} className="rounded-lg bg-muted/50 p-3">
                      <p className="text-xs text-muted-foreground">{label}</p>
                      <p className="text-sm font-medium mt-0.5">{value}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* ─── Sidebar ─── */}
          <div className="space-y-4">
            {/* Performance */}
            <Card>
              <CardHeader className="pb-2 px-4 pt-4">
                <CardTitle className="text-base flex items-center gap-2">
                  <Monitor className="h-4 w-4 text-muted-foreground" />
                  Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-xl bg-muted/40 p-3 border border-border/50">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">FPS</p>
                    <p className={`text-2xl font-bold tabular-nums ${metrics.fps >= 15 ? 'text-emerald-400' :
                      metrics.fps >= 10 ? 'text-amber-400' : 'text-red-400'
                      }`}>
                      {metrics.fps.toFixed(1)}
                    </p>
                  </div>
                  <div className="rounded-xl bg-muted/40 p-3 border border-border/50">
                    <p className="text-xs text-muted-foreground">Inference</p>
                    <p className="text-2xl font-bold tabular-nums">
                      {metrics.inferenceTime.toFixed(0)}
                      <span className="text-xs font-normal text-muted-foreground ml-0.5">ms</span>
                    </p>
                  </div>
                  <div className="rounded-xl bg-muted/40 p-3 border border-border/50">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Preprocess</p>
                    <p className="text-lg font-semibold tabular-nums">{metrics.preprocessTime.toFixed(0)}ms</p>
                  </div>
                  <div className="rounded-xl bg-muted/40 p-3 border border-border/50">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Postprocess</p>
                    <p className="text-lg font-semibold tabular-nums">{metrics.postprocessTime.toFixed(0)}ms</p>
                  </div>
                  <div className="col-span-2 rounded-xl bg-muted/40 p-3 border border-border/50">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Total Frame Time</p>
                    <p className="text-lg font-semibold tabular-nums">
                      {(metrics.preprocessTime + metrics.inferenceTime + metrics.postprocessTime).toFixed(0)}ms
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Current Detections */}
            <Card>
              <CardHeader className="pb-2 px-4 pt-4">
                <CardTitle className="text-base flex items-center gap-2">
                  <Eye className="h-4 w-4 text-muted-foreground" />
                  Detections
                  <Badge variant="secondary" className="ml-auto text-xs">{detections.length}</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <ScrollArea className="h-[180px]">
                  {detections.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-8">
                      No objects detected
                    </p>
                  ) : (
                    <div className="space-y-1.5">
                      {detections.map((det) => (
                        <div
                          key={det.id}
                          className="flex items-center justify-between p-2 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors"
                        >
                          <div className="flex items-center gap-2">
                            <div
                              className="w-2.5 h-2.5 rounded-full ring-2 ring-background"
                              style={{ backgroundColor: getClassColor(det.classId) }}
                            />
                            <span className="text-sm font-medium">{det.label}</span>
                          </div>
                          <span className="text-xs font-mono text-muted-foreground">
                            {Math.round(Math.max(0, Math.min(1, det.confidence)) * 100)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </ScrollArea>
              </CardContent>
            </Card>

            {/* Event Log */}
            <Card>
              <CardHeader className="pb-2 px-4 pt-4">
                <CardTitle className="text-base">
                  Event Log
                  <Badge variant="secondary" className="ml-2 text-xs">{stats?.totalEvents || 0}</Badge>
                </CardTitle>
                <CardDescription className="text-xs">Recent detection events</CardDescription>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <ScrollArea className="h-[200px]">
                  {eventLog.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-8">
                      No events logged yet
                    </p>
                  ) : (
                    <div className="space-y-1">
                      {eventLog.slice(-50).reverse().map((event) => (
                        <div
                          key={event.id}
                          className="flex items-center justify-between py-1.5 px-2 text-xs rounded hover:bg-muted/30 transition-colors"
                        >
                          <div className="flex items-center gap-1.5">
                            <div
                              className="w-1.5 h-1.5 rounded-full"
                              style={{ backgroundColor: getClassColor(event.classId) }}
                            />
                            <span>{event.label}</span>
                          </div>
                          <div className="flex items-center gap-2 text-muted-foreground">
                            <span className="font-mono">{Math.round(Math.max(0, Math.min(1, event.confidence)) * 100)}%</span>
                            <span>{new Date(event.timestamp).toLocaleTimeString()}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </ScrollArea>
              </CardContent>
            </Card>

            {/* Detection Summary */}
            {stats?.classCounts && Object.keys(stats.classCounts).length > 0 && (
              <Card>
                <CardHeader className="pb-2 px-4 pt-4">
                  <CardTitle className="text-base">Summary</CardTitle>
                </CardHeader>
                <CardContent className="px-4 pb-4">
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(stats.classCounts)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 10)
                      .map(([label, count]) => (
                        <Badge key={label} variant="secondary" className="text-xs">
                          {label}: {count}
                        </Badge>
                      ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Diagnostics */}
            <Card>
              <CardHeader className="pb-2 px-4 pt-4">
                <CardTitle className="text-base flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                  Diagnostics
                </CardTitle>
                <CardDescription className="text-xs">Pipeline status</CardDescription>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                <div className="space-y-1 text-xs font-mono">
                  <Row label="Provider" value={diagnostics?.provider || 'n/a'} />
                  <Row
                    label="Model"
                    value={diagnostics?.modelLoaded ? 'Loaded' : 'Not loaded'}
                    className={diagnostics?.modelLoaded ? 'text-emerald-500' : 'text-destructive'}
                  />
                  <Row label="Input" value={`${diagnostics?.inputName || '?'} [${diagnostics?.inputShape?.join(',') || '?'}]`} />
                  <Row label="Output" value={diagnostics?.outputNames?.join(', ') || '?'} />
                  <Row label="Shape" value={diagnostics?.outputShapes?.map(s => `[${s.join(',')}]`).join(' ') || '?'} />
                  <Row label="Format" value={diagnostics?.outputFormat || 'unknown'} />
                  <hr className="border-border/50 my-1.5" />
                  <Row label="Pre-thresh" value={String(diagnostics?.preThresholdCount ?? 0)} />
                  <Row label="Post-thresh" value={String(diagnostics?.postThresholdCount ?? 0)} />
                  <Row label="Post-NMS" value={String(diagnostics?.postNmsCount ?? 0)} />
                  <Row label="Inference" value={`${diagnostics?.inferenceTimeMs?.toFixed(1) ?? 0}ms`} />
                  {diagnostics?.lastError && (
                    <div className="mt-2 p-2 bg-destructive/10 rounded-lg text-destructive break-all text-[10px]">
                      {diagnostics.lastError}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-8 py-4 border-t text-center text-xs text-muted-foreground flex flex-col items-center gap-2">
          <p>
            Real-time object detection · YOLOv8n · ONNX Runtime Web · WebGPU
          </p>
          <a
            href="https://github.com/aymanaljunaid/ai-object-detection"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition-colors"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4 fill-current"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.09-.745.083-.73.083-.73 1.205.085 1.84 1.237 1.84 1.237 1.07 1.834 2.807 1.304 3.492.997.108-.775.418-1.305.762-1.605-2.665-.3-5.467-1.332-5.467-5.93 0-1.31.468-2.382 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.957-.266 1.98-.399 3-.405 1.02.006 2.047.139 3.006.405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.838 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.605-.015 2.896-.015 3.286 0 .315.21.694.825.577C20.565 21.795 24 17.295 24 12c0-6.63-5.37-12-12-12z" /></svg>
            aymanaljunaid
          </a>
        </footer>
      </div>
    </div>
  );
}

/** Small helper for diagnostics rows */
function Row({ label, value, className }: { label: string; value: string; className?: string }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-muted-foreground shrink-0">{label}</span>
      <span className={`text-right truncate ${className || ''}`}>{value}</span>
    </div>
  );
}