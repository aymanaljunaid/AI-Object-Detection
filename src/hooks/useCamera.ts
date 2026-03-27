/**
 * Camera capture hook using WebRTC/MediaDevices API
 * Provides live video stream from device camera
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type { CameraCapabilities } from '@/lib/detection/types';

export interface UseCameraOptions {
  width?: number;
  height?: number;
  frameRate?: number;
  deviceId?: string;
  facingMode?: 'user' | 'environment';
}

export interface UseCameraReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isStreaming: boolean;
  isLoading: boolean;
  error: string | null;
  capabilities: CameraCapabilities[];
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  switchCamera: (deviceId: string) => Promise<void>;
  captureFrame: () => ImageData | null;
  getVideoSize: () => { width: number; height: number };
}

export function useCamera(options: UseCameraOptions = {}): UseCameraReturn {
  const {
    width = 640,
    height = 480,
    frameRate = 30,
    deviceId,
    facingMode = 'environment'
  } = options;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [capabilities, setCapabilities] = useState<CameraCapabilities[]>([]);

  // BUG 3 FIX: enumerate cameras using the already-obtained stream permission,
  // do NOT call getUserMedia again (that would open a second leaked stream).
  const enumerateCameras = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');

      const cameraCaps: CameraCapabilities[] = videoDevices.map(device => ({
        deviceId: device.deviceId,
        label: device.label || `Camera ${device.deviceId.slice(0, 8)}`,
        resolution: { width, height },
        frameRate
      }));

      setCapabilities(cameraCaps);
      return cameraCaps;
    } catch (err) {
      console.error('Failed to enumerate cameras:', err);
      return [];
    }
  }, [width, height, frameRate]);

  // Initialize canvas for frame capture
  useEffect(() => {
    canvasRef.current = document.createElement('canvas');
    canvasRef.current.width = width;
    canvasRef.current.height = height;
    ctxRef.current = canvasRef.current.getContext('2d', { willReadFrequently: true });
  }, [width, height]);

  // Start camera stream
  const startCamera = useCallback(async () => {
    if (isStreaming || isLoading) return;

    setIsLoading(true);
    setError(null);

    try {
      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: width },
          height: { ideal: height },
          frameRate: { ideal: frameRate },
          facingMode: deviceId ? undefined : facingMode,
          deviceId: deviceId ? { exact: deviceId } : undefined
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsStreaming(true);

        // Update canvas size to match video
        if (canvasRef.current) {
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
        }
      }

      // BUG 3 FIX: enumerate cameras AFTER getting stream (permission already granted),
      // no second getUserMedia call needed inside enumerateCameras.
      await enumerateCameras();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to access camera';
      setError(errorMessage);
      console.error('Camera error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [isStreaming, isLoading, width, height, frameRate, deviceId, facingMode, enumerateCameras]);

  // Stop camera stream
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  }, []);

  // BUG 2 FIX: switchCamera now has full try/catch with error state
  const switchCamera = useCallback(async (newDeviceId: string) => {
    stopCamera();
    setIsLoading(true);
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: width },
          height: { ideal: height },
          frameRate: { ideal: frameRate },
          deviceId: { exact: newDeviceId }
        },
        audio: false
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsStreaming(true);

        if (canvasRef.current) {
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to switch camera';
      setError(errorMessage);
      console.error('Switch camera error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [stopCamera, width, height, frameRate]);

  // Capture current frame as ImageData
  const captureFrame = useCallback((): ImageData | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;

    if (!video || !canvas || !ctx || !isStreaming) {
      return null;
    }

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get image data
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }, [isStreaming]);

  // Get current video dimensions
  const getVideoSize = useCallback(() => {
    const video = videoRef.current;
    if (!video) {
      return { width, height };
    }
    return {
      width: video.videoWidth || width,
      height: video.videoHeight || height
    };
  }, [width, height]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return {
    videoRef,
    isStreaming,
    isLoading,
    error,
    capabilities,
    startCamera,
    stopCamera,
    switchCamera,
    captureFrame,
    getVideoSize
  };
}
