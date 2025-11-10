import React from "react";
import SplitText from "./SplitText";
import BlurText from "./BlurText";

function AccuracyPresentationContainer() {
  const handleAnimationCompleteSplit = () => {
    console.log("All letters have animated!");
  };

const handleAnimationCompleteBlur = () => {
  console.log('Animation completed!');
};

  return (
    <div className="accuracy-presentation-container">
      <SplitText
        text="Welcome to F1 Predictor, a machine learning powered predictor"
        className="accuracy-presentation-container-text-h1"
        delay={100}
        duration={0.3}
        ease="power3.out"
        splitType="chars"
        from={{ opacity: 0, y: 40 }}
        to={{ opacity: 1, y: 0 }}
        threshold={0.1}
        rootMargin="-100px"
        textAlign="center"
        onLetterAnimationComplete={handleAnimationCompleteSplit}
      />
    
    <BlurText
      text="Guaranteed precision of: 89%"
      delay={150}
      startDelay={5300}
      animateBy="words"
      direction="top"
      onAnimationComplete={handleAnimationCompleteBlur}
      className="accuracy-presentation-container-text-h2"
    />
      </div>
  );
}

export default AccuracyPresentationContainer;
