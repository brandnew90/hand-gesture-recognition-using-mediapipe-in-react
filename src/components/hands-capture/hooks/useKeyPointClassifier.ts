import { useEffect, useRef } from 'react';

import * as tf from '@tensorflow/tfjs';

import { cloneDeep, flatten } from 'lodash';

// MediaPipe types (loaded from CDN, defined for TypeScript)
interface Landmark {
  x: number;
  y: number;
  z: number;
}

interface Results {
  multiHandLandmarks?: Landmark[][];
  image: HTMLVideoElement | HTMLImageElement;
}

/**
 * Number of consecutive frame predictions to keep in the voting buffer.
 * The most frequent class across these frames is returned as the final label.
 * A larger window is more stable but reacts slower to genuine pose changes.
 */
const SMOOTHING_BUFFER_SIZE = 5;

/**
 * Minimum softmax probability required to emit a gesture label.
 * When the hand is moving, the model is often uncertain (low confidence).
 * Returning `null` in those frames prevents flickering labels.
 */
const CONFIDENCE_THRESHOLD = 0.6;

const calcLandmarkList = (image, landmarks) => {
  const { width: imageWidth, height: imageHeight } = image;

  const landmarkPoint: any = [];

  // Keypoint
  Object.values(landmarks).forEach((landmark: Landmark) => {
    const landmarkX = Math.min(landmark.x * imageWidth, imageWidth - 1);
    const landmarkY = Math.min(landmark.y * imageHeight, imageHeight - 1);

    landmarkPoint.push([landmarkX, landmarkY]);
  });

  return landmarkPoint;
};

const preProcessLandmark = (landmarkList) => {
  let tempLandmarkList = cloneDeep(landmarkList);

  let baseX = 0;
  let baseY = 0;

  //convert to realtive coordinates
  Object.values(tempLandmarkList).forEach((landmarkPoint, index) => {
    if (!index) {
      baseX = parseInt(landmarkPoint[0]);
      baseY = parseInt(landmarkPoint[1]);
    }

    tempLandmarkList[index][0] = tempLandmarkList[index][0] - baseX;
    tempLandmarkList[index][1] = tempLandmarkList[index][1] - baseY;
  });

  //convert to one-dimensional list
  tempLandmarkList = flatten(tempLandmarkList);

  //normalize
  const maxValue = Math.max(
    ...tempLandmarkList.map((value) => Math.abs(value))
  );
  tempLandmarkList = tempLandmarkList.map((value) => value / maxValue);
  return tempLandmarkList;
};

function useKeyPointClassifier() {
  const model = useRef<any>();
  const smoothingBuffer = useRef<number[]>([]);

  /**
   * Run the model and return { classId, confidence }.
   * Returns null when the model is not yet loaded.
   */
  const keyPointClassifier = async (landmarkList): Promise<{ classId: number; confidence: number } | null> => {
    if (!model.current) return null;

    const inputTensor = tf.tensor2d([landmarkList]);
    const outputTensor = model.current.execute(inputTensor).squeeze();
    const scores: Float32Array = await outputTensor.data();
    inputTensor.dispose();
    outputTensor.dispose();

    let classId = 0;
    let maxScore = scores[0];
    for (let i = 1; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        classId = i;
      }
    }
    return { classId, confidence: maxScore };
  };

  /**
   * Apply a majority-vote smoothing buffer over recent predictions.
   * Returns the most frequent classId seen in the last SMOOTHING_BUFFER_SIZE
   * frames, which stabilises the output when the hand is moving.
   */
  const smoothedClassId = (classId: number): number => {
    smoothingBuffer.current.push(classId);
    if (smoothingBuffer.current.length > SMOOTHING_BUFFER_SIZE) {
      smoothingBuffer.current.shift();
    }

    // Vote: return the most frequent class in the buffer
    const counts: Record<number, number> = {};
    for (const id of smoothingBuffer.current) {
      counts[id] = (counts[id] ?? 0) + 1;
    }
    return Number(
      Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]
    );
  };

  /**
   * Process a single hand's landmarks and return a gesture class ID.
   * Returns -1 when confidence is below CONFIDENCE_THRESHOLD (e.g. while
   * the hand is mid-motion and the pose is ambiguous).
   */
  const processLandmark = async (handLandmarks: Results, image): Promise<number> => {
    const landmarkList = calcLandmarkList(image, handLandmarks);
    const preProcessedLandmarkList = preProcessLandmark(landmarkList);
    const result = await keyPointClassifier(preProcessedLandmarkList);
    if (!result || result.confidence < CONFIDENCE_THRESHOLD) {
      // Below threshold — clear the buffer so stale votes don't linger
      smoothingBuffer.current = [];
      return -1;
    }
    return smoothedClassId(result.classId);
  };


  useEffect(() => {
    (async function loadModel () {
      model.current = await tf.loadGraphModel(
        `/tf-models/key-point-classifier/model.json`
      );
    })()
  }, []);

  return { processLandmark };
}

export default useKeyPointClassifier;