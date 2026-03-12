/**
 * Event Logger Utility
 * Persists detection events to localStorage with automatic cleanup
 */

import type { DetectionEvent, DetectionSession } from './types';
import { generateId } from './types';

const STORAGE_KEY_PREFIX = 'object_detection_';
const EVENTS_KEY = `${STORAGE_KEY_PREFIX}events`;
const SESSIONS_KEY = `${STORAGE_KEY_PREFIX}sessions`;
const MAX_EVENTS = 1000; // Maximum events to store
const MAX_SESSIONS = 50; // Maximum sessions to store

/**
 * Save detection events to localStorage
 */
export function saveEvents(events: DetectionEvent[]): void {
  if (typeof window === 'undefined') return;

  try {
    // Get existing events
    const existing = getEvents();
    
    // Merge and limit
    const merged = [...existing, ...events].slice(-MAX_EVENTS);
    
    localStorage.setItem(EVENTS_KEY, JSON.stringify(merged));
  } catch (error) {
    console.error('Failed to save events:', error);
  }
}

/**
 * Get all stored events
 */
export function getEvents(): DetectionEvent[] {
  if (typeof window === 'undefined') return [];

  try {
    const stored = localStorage.getItem(EVENTS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

/**
 * Get events filtered by criteria
 */
export function getFilteredEvents(options: {
  sessionId?: string;
  label?: string;
  minConfidence?: number;
  startTime?: number;
  endTime?: number;
  limit?: number;
}): DetectionEvent[] {
  let events = getEvents();

  if (options.sessionId) {
    events = events.filter(e => e.sessionId === options.sessionId);
  }

  if (options.label) {
    events = events.filter(e => e.label === options.label);
  }

  if (options.minConfidence !== undefined) {
    events = events.filter(e => e.confidence >= options.minConfidence);
  }

  if (options.startTime !== undefined) {
    events = events.filter(e => e.timestamp >= options.startTime!);
  }

  if (options.endTime !== undefined) {
    events = events.filter(e => e.timestamp <= options.endTime!);
  }

  // Sort by timestamp (newest first)
  events.sort((a, b) => b.timestamp - a.timestamp);

  if (options.limit !== undefined) {
    events = events.slice(0, options.limit);
  }

  return events;
}

/**
 * Clear all events
 */
export function clearEvents(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(EVENTS_KEY);
}

/**
 * Create and save a new detection session
 */
export function createSession(settings: DetectionSession['settings']): DetectionSession {
  const session: DetectionSession = {
    id: generateId(),
    startTime: Date.now(),
    totalDetections: 0,
    settings
  };

  saveSession(session);
  return session;
}

/**
 * Update an existing session
 */
export function updateSession(sessionId: string, updates: Partial<DetectionSession>): void {
  if (typeof window === 'undefined') return;

  const sessions = getSessions();
  const index = sessions.findIndex(s => s.id === sessionId);

  if (index !== -1) {
    sessions[index] = { ...sessions[index], ...updates };
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
  }
}

/**
 * Save a session
 */
function saveSession(session: DetectionSession): void {
  if (typeof window === 'undefined') return;

  try {
    const sessions = getSessions();
    sessions.push(session);
    
    // Keep only the most recent sessions
    const trimmed = sessions.slice(-MAX_SESSIONS);
    
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(trimmed));
  } catch (error) {
    console.error('Failed to save session:', error);
  }
}

/**
 * Get all sessions
 */
export function getSessions(): DetectionSession[] {
  if (typeof window === 'undefined') return [];

  try {
    const stored = localStorage.getItem(SESSIONS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

/**
 * Get a specific session
 */
export function getSession(sessionId: string): DetectionSession | null {
  const sessions = getSessions();
  return sessions.find(s => s.id === sessionId) || null;
}

/**
 * Get session statistics
 */
export function getSessionStats(sessionId: string): {
  totalDetections: number;
  uniqueClasses: string[];
  averageConfidence: number;
  detectionRate: number; // detections per second
} | null {
  const session = getSession(sessionId);
  if (!session) return null;

  const events = getFilteredEvents({ sessionId });
  
  if (events.length === 0) {
    return {
      totalDetections: 0,
      uniqueClasses: [],
      averageConfidence: 0,
      detectionRate: 0
    };
  }

  const uniqueClasses = [...new Set(events.map(e => e.label))];
  const avgConfidence = events.reduce((sum, e) => sum + e.confidence, 0) / events.length;
  
  const duration = ((session.endTime || Date.now()) - session.startTime) / 1000;
  const detectionRate = events.length / duration;

  return {
    totalDetections: events.length,
    uniqueClasses,
    averageConfidence: avgConfidence,
    detectionRate
  };
}

/**
 * Export events as JSON
 */
export function exportEvents(format: 'json' | 'csv' = 'json'): string {
  const events = getEvents();

  if (format === 'json') {
    return JSON.stringify(events, null, 2);
  }

  // CSV format
  const headers = ['id', 'label', 'classId', 'confidence', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'timestamp', 'sessionId'];
  const rows = events.map(e => [
    e.id,
    e.label,
    e.classId,
    e.confidence.toFixed(4),
    e.bbox.x.toFixed(2),
    e.bbox.y.toFixed(2),
    e.bbox.width.toFixed(2),
    e.bbox.height.toFixed(2),
    e.timestamp,
    e.sessionId
  ].join(','));

  return [headers.join(','), ...rows].join('\n');
}

/**
 * Get summary statistics
 */
export function getSummary(): {
  totalEvents: number;
  totalSessions: number;
  classCounts: Record<string, number>;
  oldestEvent: number | null;
  newestEvent: number | null;
} {
  const events = getEvents();
  const sessions = getSessions();

  const classCounts: Record<string, number> = {};
  
  for (const event of events) {
    classCounts[event.label] = (classCounts[event.label] || 0) + 1;
  }

  return {
    totalEvents: events.length,
    totalSessions: sessions.length,
    classCounts,
    oldestEvent: events.length > 0 ? events[0].timestamp : null,
    newestEvent: events.length > 0 ? events[events.length - 1].timestamp : null
  };
}
