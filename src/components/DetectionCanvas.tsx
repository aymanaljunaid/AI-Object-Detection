'use client';

/**
 * Detection Canvas Overlay Component
 * Renders bounding boxes, labels, and confidence scores on a canvas
 */

import React, { useRef, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react';
import type { Detection, PerformanceMetrics } from '@/lib/detection/types';
import { getClassColor } from '@/lib/detection/types';

export interface DetectionCanvasProps {
  detections: Detection[];
  width: number;
  height: number;
  showLabels?: boolean;
  showConfidence?: boolean;
  showFps?: boolean;
  metrics?: PerformanceMetrics;
  className?: string;
}

export interface DetectionCanvasRef {
  getCanvas: () => HTMLCanvasElement | null;
  clear: () => void;
}

export const DetectionCanvas = forwardRef<DetectionCanvasRef, DetectionCanvasProps>(
  function DetectionCanvas(
    {
      detections,
      width,
      height,
      showLabels = true,
      showConfidence = true,
      showFps = true,
      metrics,
      className = ''
    },
    ref
  ) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Expose methods via ref
    useImperativeHandle(ref, () => ({
      getCanvas: () => canvasRef.current,
      clear: () => {
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
          }
        }
      }
    }));

    // Draw a single detection
    const drawDetection = useCallback((
      ctx: CanvasRenderingContext2D,
      detection: Detection
    ) => {
      const { bbox, label, classId, confidence } = detection;
      const color = getClassColor(classId);

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

      // Draw filled corners for better visibility
      const cornerSize = 8;
      ctx.fillStyle = color;
      
      // Top-left corner
      ctx.fillRect(bbox.x - 1, bbox.y - 1, cornerSize, 3);
      ctx.fillRect(bbox.x - 1, bbox.y - 1, 3, cornerSize);
      
      // Top-right corner
      ctx.fillRect(bbox.x + bbox.width - cornerSize + 1, bbox.y - 1, cornerSize, 3);
      ctx.fillRect(bbox.x + bbox.width - 2, bbox.y - 1, 3, cornerSize);
      
      // Bottom-left corner
      ctx.fillRect(bbox.x - 1, bbox.y + bbox.height - 2, cornerSize, 3);
      ctx.fillRect(bbox.x - 1, bbox.y + bbox.height - cornerSize + 1, 3, cornerSize);
      
      // Bottom-right corner
      ctx.fillRect(bbox.x + bbox.width - cornerSize + 1, bbox.y + bbox.height - 2, cornerSize, 3);
      ctx.fillRect(bbox.x + bbox.width - 2, bbox.y + bbox.height - cornerSize + 1, 3, cornerSize);

      // Draw label background and text
      if (showLabels || showConfidence) {
        const labelText = showLabels && showConfidence
          ? `${label} ${(confidence * 100).toFixed(0)}%`
          : showLabels
            ? label
            : `${(confidence * 100).toFixed(0)}%`;

        ctx.font = 'bold 12px system-ui, -apple-system, sans-serif';
        const textWidth = ctx.measureText(labelText).width;
        const textHeight = 16;
        const padding = 4;

        // Position label above the box (or inside if at top edge)
        const labelY = bbox.y > textHeight + padding 
          ? bbox.y - textHeight - padding 
          : bbox.y + padding;
        const labelX = bbox.x;

        // Draw label background
        ctx.fillStyle = color;
        ctx.fillRect(labelX, labelY, textWidth + padding * 2, textHeight);

        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, labelX + padding, labelY + textHeight / 2);
      }
    }, [showLabels, showConfidence]);

    // Draw FPS and metrics overlay
    const drawMetrics = useCallback((
      ctx: CanvasRenderingContext2D,
      metrics: PerformanceMetrics
    ) => {
      const padding = 10;
      const lineHeight = 20;
      let y = padding;

      ctx.font = 'bold 12px system-ui, -apple-system, sans-serif';
      ctx.textBaseline = 'top';

      // FPS
      ctx.fillStyle = metrics.fps >= 15 ? '#22c55e' : metrics.fps >= 10 ? '#eab308' : '#ef4444';
      ctx.fillText(`FPS: ${metrics.fps.toFixed(1)}`, padding, y);
      y += lineHeight;

      // Inference time
      ctx.fillStyle = '#ffffff';
      ctx.fillText(`Inference: ${metrics.inferenceTime.toFixed(1)}ms`, padding, y);
      y += lineHeight;

      // Total frame time
      const totalTime = metrics.preprocessTime + metrics.inferenceTime + metrics.postprocessTime;
      ctx.fillStyle = '#9ca3af';
      ctx.fillText(`Total: ${totalTime.toFixed(1)}ms`, padding, y);
    }, []);

    // Main render effect
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Draw all detections
      for (const detection of detections) {
        drawDetection(ctx, detection);
      }

      // Draw metrics overlay
      if (showFps && metrics) {
        drawMetrics(ctx, metrics);
      }
    }, [detections, width, height, showFps, metrics, drawDetection, drawMetrics]);

    return (
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className={`absolute top-0 left-0 pointer-events-none ${className}`}
      />
    );
  }
);

export default DetectionCanvas;
