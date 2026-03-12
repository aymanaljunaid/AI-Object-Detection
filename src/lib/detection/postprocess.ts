/**
 * Postprocessing utilities for object detection
 * Implements Non-Maximum Suppression (NMS) and other filtering operations
 */

import type { Detection, BoundingBox } from './types';

/**
 * Calculate Intersection over Union (IoU) between two bounding boxes
 */
export function calculateIoU(box1: BoundingBox, box2: BoundingBox): number {
  // Calculate intersection coordinates
  const x1 = Math.max(box1.x, box2.x);
  const y1 = Math.max(box1.y, box2.y);
  const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
  const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

  // Calculate intersection area
  const intersectionWidth = Math.max(0, x2 - x1);
  const intersectionHeight = Math.max(0, y2 - y1);
  const intersectionArea = intersectionWidth * intersectionHeight;

  // Calculate union area
  const box1Area = box1.width * box1.height;
  const box2Area = box2.width * box2.height;
  const unionArea = box1Area + box2Area - intersectionArea;

  // Avoid division by zero
  if (unionArea === 0) return 0;

  return intersectionArea / unionArea;
}

/**
 * Non-Maximum Suppression (NMS)
 * Removes overlapping detections, keeping the ones with highest confidence
 * 
 * @param detections - Array of detections to filter
 * @param iouThreshold - IoU threshold for considering boxes as overlapping
 * @param scoreThreshold - Minimum confidence score to keep
 * @returns Filtered detections after NMS
 */
export function applyNMS(
  detections: Detection[],
  iouThreshold: number = 0.45,
  scoreThreshold: number = 0.25
): Detection[] {
  if (detections.length === 0) return [];

  // Filter by score threshold first
  const filteredDetections = detections.filter(d => d.confidence >= scoreThreshold);

  if (filteredDetections.length === 0) return [];

  // Sort by confidence (descending)
  const sortedDetections = [...filteredDetections].sort(
    (a, b) => b.confidence - a.confidence
  );

  const keptDetections: Detection[] = [];
  const suppressed = new Set<string>();

  for (let i = 0; i < sortedDetections.length; i++) {
    const current = sortedDetections[i];
    
    // Skip if already suppressed
    if (suppressed.has(current.id)) continue;

    keptDetections.push(current);

    // Suppress overlapping boxes of the same class
    for (let j = i + 1; j < sortedDetections.length; j++) {
      const other = sortedDetections[j];
      
      if (suppressed.has(other.id)) continue;

      // Only apply NMS for same class
      if (current.classId === other.classId) {
        const iou = calculateIoU(current.bbox, other.bbox);
        
        if (iou > iouThreshold) {
          suppressed.add(other.id);
        }
      }
    }
  }

  return keptDetections;
}

/**
 * Class-aware NMS (applies NMS per class independently)
 * This can produce better results for multi-class scenarios
 */
export function applyClassAwareNMS(
  detections: Detection[],
  iouThreshold: number = 0.45,
  scoreThreshold: number = 0.25
): Detection[] {
  if (detections.length === 0) return [];

  // Group by class
  const classGroups = new Map<number, Detection[]>();
  
  for (const det of detections) {
    if (det.confidence < scoreThreshold) continue;
    
    const group = classGroups.get(det.classId) || [];
    group.push(det);
    classGroups.set(det.classId, group);
  }

  // Apply NMS per class
  const allResults: Detection[] = [];

  for (const [, classDetections] of classGroups) {
    const nmsResults = applyNMS(classDetections, iouThreshold, 0); // Already filtered
    allResults.push(...nmsResults);
  }

  // Sort by confidence
  return allResults.sort((a, b) => b.confidence - a.confidence);
}

/**
 * Soft-NMS implementation
 * Instead of removing boxes, reduces their confidence based on IoU
 */
export function applySoftNMS(
  detections: Detection[],
  iouThreshold: number = 0.5,
  sigma: number = 0.5,
  scoreThreshold: number = 0.25
): Detection[] {
  if (detections.length === 0) return [];

  // Clone detections to avoid mutation
  const workingDetections = detections.map(d => ({
    ...d,
    confidence: d.confidence
  }));

  const result: Detection[] = [];

  while (workingDetections.length > 0) {
    // Find max score detection
    let maxIdx = 0;
    let maxScore = workingDetections[0].confidence;

    for (let i = 1; i < workingDetections.length; i++) {
      if (workingDetections[i].confidence > maxScore) {
        maxScore = workingDetections[i].confidence;
        maxIdx = i;
      }
    }

    const maxDet = workingDetections[maxIdx];
    
    if (maxDet.confidence < scoreThreshold) break;

    result.push(maxDet);
    workingDetections.splice(maxIdx, 1);

    // Update remaining scores
    for (let i = workingDetections.length - 1; i >= 0; i--) {
      const det = workingDetections[i];
      
      // Only for same class
      if (det.classId === maxDet.classId) {
        const iou = calculateIoU(maxDet.bbox, det.bbox);
        
        if (iou > iouThreshold) {
          // Gaussian penalty
          det.confidence *= Math.exp(-(iou * iou) / sigma);
        }
      }
    }
  }

  return result;
}

/**
 * Filter detections by minimum size
 */
export function filterBySize(
  detections: Detection[],
  minSize: number = 10,
  maxSize: number = Infinity
): Detection[] {
  return detections.filter(d => {
    const size = Math.max(d.bbox.width, d.bbox.height);
    return size >= minSize && size <= maxSize;
  });
}

/**
 * Filter detections by region of interest
 */
export function filterByROI(
  detections: Detection[],
  roi: BoundingBox
): Detection[] {
  return detections.filter(d => {
    // Check if detection center is within ROI
    const centerX = d.bbox.x + d.bbox.width / 2;
    const centerY = d.bbox.y + d.bbox.height / 2;
    
    return (
      centerX >= roi.x &&
      centerX <= roi.x + roi.width &&
      centerY >= roi.y &&
      centerY <= roi.y + roi.height
    );
  });
}

/**
 * Merge similar detections across frames (simple temporal smoothing)
 */
export function mergeDetections(
  currentDetections: Detection[],
  previousDetections: Detection[],
  iouThreshold: number = 0.3
): Detection[] {
  if (previousDetections.length === 0) return currentDetections;
  if (currentDetections.length === 0) return [];

  const result: Detection[] = [];
  const matchedPrev = new Set<string>();

  for (const curr of currentDetections) {
    let bestMatch: Detection | null = null;
    let bestIoU = 0;

    for (const prev of previousDetections) {
      if (matchedPrev.has(prev.id)) continue;
      if (prev.classId !== curr.classId) continue;

      const iou = calculateIoU(curr.bbox, prev.bbox);
      if (iou > iouThreshold && iou > bestIoU) {
        bestIoU = iou;
        bestMatch = prev;
      }
    }

    if (bestMatch) {
      // Smoothed detection (weighted average)
      const weight = 0.7; // Favor current detection
      result.push({
        ...curr,
        bbox: {
          x: bestMatch.bbox.x * (1 - weight) + curr.bbox.x * weight,
          y: bestMatch.bbox.y * (1 - weight) + curr.bbox.y * weight,
          width: bestMatch.bbox.width * (1 - weight) + curr.bbox.width * weight,
          height: bestMatch.bbox.height * (1 - weight) + curr.bbox.height * weight
        },
        confidence: Math.max(bestMatch.confidence, curr.confidence)
      });
      matchedPrev.add(bestMatch.id);
    } else {
      result.push(curr);
    }
  }

  return result;
}
