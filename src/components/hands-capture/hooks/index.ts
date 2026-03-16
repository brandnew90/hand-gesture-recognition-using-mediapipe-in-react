import { Ref, useEffect, useRef } from 'react';
import { Camera } from '@mediapipe/camera_utils';
import {
  drawConnectors,
  drawLandmarks,
  drawRectangle,
} from '@mediapipe/drawing_utils';
import { Hands, HAND_CONNECTIONS } from '@mediapipe/hands';
import useKeyPointClassifier from '../hooks/useKeyPointClassifier';
import useSwipeDetector, { SwipeDirection } from '../hooks/useSwipeDetector';
import CONFIGS from '../../../../constants';

const maxVideoWidth = 960;
const maxVideoHeight = 540;

/** Mapping from swipe direction to a Unicode arrow for compact display. */
const SWIPE_ARROW: Record<SwipeDirection, string> = {
  'None': '',
  'Swipe Left': '← Swipe Left',
  'Swipe Right': '→ Swipe Right',
  'Swipe Up': '↑ Swipe Up',
  'Swipe Down': '↓ Swipe Down',
};

interface IHandGestureLogic {
  videoElement: Ref<any>
  canvasEl: Ref<any>
}

function useGestureRecognition({videoElement, canvasEl}: IHandGestureLogic) {
  const hands = useRef<any>(null);
  const camera = useRef<any>(null);
  const handsGesture = useRef<any>([]);
  const swipeDirections = useRef<SwipeDirection[]>([]);

  const { processLandmark } = useKeyPointClassifier();
  const { detectSwipe, resetHistory } = useSwipeDetector();

  async function onResults(results) {
    if (canvasEl.current) {
      const ctx = canvasEl.current.getContext('2d');

      ctx.save();
      ctx.clearRect(0, 0, canvasEl.current.width, canvasEl.current.height);
      ctx.drawImage(results.image, 0, 0, maxVideoWidth, maxVideoHeight);

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        // Runs once for every hand
        for (const [index, landmarks] of results.multiHandLandmarks.entries()) {
          // Static pose classification
          processLandmark(landmarks, results.image).then(
            (val) => (handsGesture.current[index] = val)
          );

          // Dynamic swipe detection
          swipeDirections.current[index] = detectSwipe(landmarks);

          const landmarksX = landmarks.map((landmark) => landmark.x);
          const landmarksY = landmarks.map((landmark) => landmark.y);

          // Draw static gesture label (pose) above the bounding box
          ctx.fillStyle = '#ff0000';
          ctx.font = '24px serif';
          ctx.fillText(
            CONFIGS.keypointClassifierLabels[handsGesture.current[index]] ?? '',
            maxVideoWidth * Math.min(...landmarksX),
            maxVideoHeight * Math.min(...landmarksY) - 15
          );

          drawRectangle(
            ctx,
            {
              xCenter:
                Math.min(...landmarksX) +
                (Math.max(...landmarksX) - Math.min(...landmarksX)) / 2,
              yCenter:
                Math.min(...landmarksY) +
                (Math.max(...landmarksY) - Math.min(...landmarksY)) / 2,
              width: Math.max(...landmarksX) - Math.min(...landmarksX),
              height: Math.max(...landmarksY) - Math.min(...landmarksY),
              rotation: 0,
            },
            {
              fillColor: 'transparent',
              color: '#ff0000',
              lineWidth: 1,
            }
          );
          drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
            color: '#00ffff',
            lineWidth: 2,
          });
          drawLandmarks(ctx, landmarks, {
            color: '#ffff29',
            lineWidth: 1,
          });
        }

        // Draw swipe direction overlay in the top-left corner
        const currentSwipe: SwipeDirection = swipeDirections.current[0] ?? 'None';
        const swipeLabel = SWIPE_ARROW[currentSwipe] ?? '';
        if (swipeLabel) {
          ctx.font = 'bold 36px serif';
          ctx.fillStyle = 'rgba(0,0,0,0.45)';
          ctx.fillRect(8, 8, ctx.measureText(swipeLabel).width + 16, 48);
          ctx.fillStyle = '#00ff99';
          ctx.fillText(swipeLabel, 16, 44);
        }
      } else {
        // No hands detected — reset swipe history so the next detection
        // starts from a clean state.
        resetHistory();
      }

      ctx.restore();
    }
  }

  const loadHands = () => {
    hands.current = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    hands.current.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    hands.current.onResults(onResults);
  };

  useEffect(() => {
    (async function initCamara() {
      camera.current = new Camera(videoElement.current, {
        onFrame: async () => {
          await hands.current.send({ image: videoElement.current });
        },
        width: maxVideoWidth,
        height: maxVideoHeight,
      });
      camera.current.start();
    })();

    loadHands();
  }, []);

  return { maxVideoHeight, maxVideoWidth, canvasEl, videoElement };
}

export default useGestureRecognition;
