const prefix = 'http://localhost:3000';

export { prefix };

const CONFIGS = {
  /** Labels for the static hand-pose (keypoint) classifier. */
  keypointClassifierLabels: ['Open', 'Closed', 'Pointing'],

  /**
   * Labels for the point-history (dynamic motion) classifier.
   * These match the output classes produced by the training scripts in
   * the training/ directory.  Index 0 is always "No Swipe" so that
   * the rule-based swipe detector and the ML-based classifier share the
   * same label set.
   */
  pointHistoryClassifierLabels: [
    'No Swipe',
    'Swipe Left',
    'Swipe Right',
    'Swipe Up',
    'Swipe Down',
  ],

  /**
   * Number of past frames stored in the point-history window.
   * Must match HISTORY_LENGTH in useSwipeDetector.ts and the value
   * used when collecting training data (training/collect_data.py).
   */
  pointHistoryLength: 16,
};

export default CONFIGS;
