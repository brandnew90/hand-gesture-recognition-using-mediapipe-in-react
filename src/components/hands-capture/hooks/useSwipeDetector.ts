import { useCallback, useRef } from 'react';

export type SwipeDirection = 'None' | 'Swipe Left' | 'Swipe Right' | 'Swipe Up' | 'Swipe Down';

/** Minimal shape of a MediaPipe hand landmark required by this hook. */
interface Landmark {
  x: number;
  y: number;
  z?: number;
}

// Number of frames of position history used to detect a swipe
const HISTORY_LENGTH = 16;

// Minimum fraction of screen width/height that qualifies as a swipe
const SWIPE_THRESHOLD = 0.15;

// Frames to wait after a swipe before detecting the next one
const COOLDOWN_FRAMES = 30;

interface Point {
  x: number;
  y: number;
}

/**
 * useSwipeDetector
 *
 * Rule-based hook that tracks the index finger tip (landmark #8) over
 * HISTORY_LENGTH frames and detects directional swipes without requiring
 * any ML model retraining.
 *
 * To train an ML model for swipe gestures instead, see the training/
 * directory at the root of this repository.
 */
function useSwipeDetector() {
  const pointHistory = useRef<Point[]>([]);
  const cooldownCounter = useRef<number>(0);
  const lastSwipe = useRef<SwipeDirection>('None');

  /**
   * Call once per frame for each detected hand.
   * @param landmarks  Array of 21 MediaPipe hand landmarks (each has .x and .y in [0,1]).
   * @returns The detected SwipeDirection, or 'None'.
   */
  const detectSwipe = useCallback((landmarks: Landmark[]): SwipeDirection => {
    // Drain cooldown
    if (cooldownCounter.current > 0) {
      cooldownCounter.current -= 1;
      // Keep reporting the last swipe during cooldown so the label stays visible
      return lastSwipe.current;
    }

    // Index finger tip is landmark #8
    const indexTip = landmarks[8];
    const point: Point = { x: indexTip.x, y: indexTip.y };

    pointHistory.current.push(point);
    if (pointHistory.current.length > HISTORY_LENGTH) {
      pointHistory.current.shift();
    }

    // Need a full window before analysing
    if (pointHistory.current.length < HISTORY_LENGTH) {
      return 'None';
    }

    const oldest = pointHistory.current[0];
    const newest = pointHistory.current[pointHistory.current.length - 1];

    const dx = newest.x - oldest.x;
    const dy = newest.y - oldest.y;
    const absDx = Math.abs(dx);
    const absDy = Math.abs(dy);

    if (absDx < SWIPE_THRESHOLD && absDy < SWIPE_THRESHOLD) {
      lastSwipe.current = 'None';
      return 'None';
    }

    let direction: SwipeDirection;
    if (absDx >= absDy) {
      direction = dx > 0 ? 'Swipe Right' : 'Swipe Left';
    } else {
      direction = dy > 0 ? 'Swipe Down' : 'Swipe Up';
    }

    cooldownCounter.current = COOLDOWN_FRAMES;
    lastSwipe.current = direction;
    // Clear history so the next gesture starts fresh
    pointHistory.current = [];
    return direction;
  }, []);

  /** Call when the hand disappears from the frame. */
  const resetHistory = useCallback(() => {
    pointHistory.current = [];
    lastSwipe.current = 'None';
  }, []);

  return { detectSwipe, resetHistory };
}

export default useSwipeDetector;
