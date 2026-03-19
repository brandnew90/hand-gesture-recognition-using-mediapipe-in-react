import { Ref, useEffect, useRef } from 'react';
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
  videoElement: Ref<HTMLVideoElement>
  canvasEl: Ref<HTMLCanvasElement>
  isReady: boolean
}

function useGestureRecognition({videoElement, canvasEl, isReady}: IHandGestureLogic) {
  const hands = useRef<any>(null);
  const camera = useRef<any>(null);
  const handsGesture = useRef<any>([]);
  const swipeDirections = useRef<SwipeDirection[]>([]);
  const mediapipeModules = useRef<any>({});

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

          if (mediapipeModules.current.drawingUtils) {
            mediapipeModules.current.drawingUtils.drawRectangle(
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
            mediapipeModules.current.drawingUtils.drawConnectors(ctx, landmarks, mediapipeModules.current.HAND_CONNECTIONS, {
              color: '#00ffff',
              lineWidth: 2,
            });
            mediapipeModules.current.drawingUtils.drawLandmarks(ctx, landmarks, {
              color: '#ffff29',
              lineWidth: 1,
            });
          }
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

  const loadHands = (HandsClass: any, HAND_CONNECTIONS: any) => {
    mediapipeModules.current.HAND_CONNECTIONS = HAND_CONNECTIONS;
    
    hands.current = new HandsClass({
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
    if (!isReady || !videoElement.current || !canvasEl.current) {
      return;
    }

    const initMediaPipe = async () => {
      try {
        console.log('Starting MediaPipe initialization...');
        console.log('Video element:', videoElement.current);
        console.log('Canvas element:', canvasEl.current);
        
        // Check if camera devices are available
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        console.log('Available video devices:', videoDevices);
        
        if (videoDevices.length === 0) {
          throw new Error('No camera devices found');
        }
        
        // Test direct camera access first
        console.log('Testing direct camera access...');
        const testStream = await navigator.mediaDevices.getUserMedia({ video: true });
        console.log('Direct camera access successful, stream:', testStream);
        testStream.getTracks().forEach(track => track.stop());
        console.log('Test stream stopped');
        
        // Small delay to ensure DOM is fully ready
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // Wait for MediaPipe libraries to load from CDN
        while (!(window as any).Hands || !(window as any).Camera) {
          console.log('Waiting for MediaPipe to load...');
          await new Promise(resolve => setTimeout(resolve, 100));
        }

        console.log('MediaPipe loaded');

        const Camera = (window as any).Camera;
        const Hands = (window as any).Hands;
        const HAND_CONNECTIONS = (window as any).HAND_CONNECTIONS;
        
        mediapipeModules.current.drawingUtils = (window as any).drawingUtils || (window as any);

        loadHands(Hands, HAND_CONNECTIONS);
        console.log('Hands model loaded');

        if (!videoElement.current) {
          throw new Error('Video element is null');
        }

        console.log('Creating Camera instance with video element:', videoElement.current);
        console.log('Video element parent:', videoElement.current.parentElement);
        console.log('Video element in document:', document.body.contains(videoElement.current));
        
        camera.current = new Camera(videoElement.current, {
          onFrame: async () => {
            if (hands.current && videoElement.current) {
              await hands.current.send({ image: videoElement.current });
            }
          },
          width: maxVideoWidth,
          height: maxVideoHeight,
        });
        
        console.log('Starting camera...');
        camera.current.start();
        console.log('Camera started successfully');
      } catch (error) {
        console.error('Failed to acquire camera feed:', error);
        console.error('Error details:', error.message, error.stack);
      }
    };

    initMediaPipe();
  }, [isReady]);

  return { maxVideoHeight, maxVideoWidth, canvasEl, videoElement };
}

export default useGestureRecognition;
