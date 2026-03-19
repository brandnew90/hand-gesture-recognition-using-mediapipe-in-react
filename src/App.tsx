import { useRef, useEffect, useState } from "react";
import useGestureRecognition from "./components/hands-capture/hooks";

function App() {
  const videoElement = useRef<HTMLVideoElement>(null)
  const canvasEl = useRef<HTMLCanvasElement>(null)
  const [isReady, setIsReady] = useState(false);
  
  useEffect(() => {
    // Ensure refs are set before initializing MediaPipe
    if (videoElement.current && canvasEl.current) {
      setIsReady(true);
    }
  }, []);
  
  const { maxVideoWidth, maxVideoHeight } = useGestureRecognition({
    videoElement,
    canvasEl,
    isReady
  });

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        flexDirection: 'column',
      }}
    >
      <video
        style={{ width: maxVideoWidth, height: maxVideoHeight }}
        className='video'
        width={maxVideoWidth}
        height={maxVideoHeight}
        playsInline
        ref={videoElement}
      />
      <canvas 
        ref={canvasEl} 
        width={maxVideoWidth} 
        height={maxVideoHeight} 
        />
    </div>
  );
}

export default App