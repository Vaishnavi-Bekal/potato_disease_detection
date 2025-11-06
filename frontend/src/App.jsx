import React from "react";
import ImageUploader from "./components/ImageUploader";

function App() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-green-100 to-green-300 p-8">
      <h1 className="text-3xl font-bold mb-6 text-green-800">
        🍃 Potato Leaf Disease Detector
      </h1>
      <ImageUploader />
    </div>
  );
}

export default App;
