/**
 * Types for AI Real-Time Object Detection MVP
 */

// COCO dataset class labels (80 classes)
export const COCO_CLASSES: readonly string[] = [
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush'
] as const;

export type CocoClass = typeof COCO_CLASSES[number];

// Bounding box in [x, y, width, height] format (pixel coordinates)
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

// Detection result from the model
export interface Detection {
  id: string;
  bbox: BoundingBox;
  label: CocoClass;
  classId: number;
  confidence: number;
  timestamp: number;
}

// Performance metrics
export interface PerformanceMetrics {
  fps: number;
  inferenceTime: number;  // ms
  preprocessTime: number; // ms
  postprocessTime: number; // ms
  frameCount: number;
}

// Detection event for logging
export interface DetectionEvent {
  id: string;
  label: CocoClass;
  classId: number;
  confidence: number;
  bbox: BoundingBox;
  timestamp: number;
  sessionId: string;
}

// Detection session
export interface DetectionSession {
  id: string;
  startTime: number;
  endTime?: number;
  totalDetections: number;
  settings: DetectionSettings;
}

// User-configurable detection settings
export interface DetectionSettings {
  confidenceThreshold: number;
  iouThreshold: number;
  maxDetections: number;
  inputWidth: number;
  inputHeight: number;
}

// Default settings optimized for real-time performance
export const DEFAULT_SETTINGS: DetectionSettings = {
  confidenceThreshold: 0.5,
  iouThreshold: 0.45,
  maxDetections: 100,
  inputWidth: 640,
  inputHeight: 640,
};

// Model metadata
export interface ModelInfo {
  name: string;
  version: string;
  inputShape: [number, number, number, number]; // [batch, channels, height, width]
  numClasses: number;
  fileSize: number;
}

// Camera capabilities
export interface CameraCapabilities {
  deviceId: string;
  label: string;
  resolution: { width: number; height: number };
  frameRate: number;
}

// Color palette for bounding boxes (distinct colors for different classes)
export const CLASS_COLORS: readonly string[] = [
  '#FF6B6B',
  '#4ECDC4',
  '#45B7D1',
  '#96CEB4',
  '#FFEAA7',
  '#DDA0DD',
  '#98D8C8',
  '#F7DC6F',
  '#BB8FCE',
  '#85C1E9',
  '#F8B500',
  '#00CED1',
  '#FF69B4',
  '#32CD32',
  '#FFD700',
  '#FF4500',
  '#1E90FF',
  '#00FA9A',
  '#FF1493',
  '#00BFFF',
] as const;

// Get color for a class ID
export function getClassColor(classId: number): string {
  return CLASS_COLORS[classId % CLASS_COLORS.length];
}

// Generate unique ID
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}