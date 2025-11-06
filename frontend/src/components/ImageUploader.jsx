import React, { useState } from "react";
import axios from "axios";

const ImageUploader = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handle file selection
  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
  };

  // Handle upload + prediction
  const handleUpload = async () => {
    if (!file) return alert("Please select an image first!");
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const apiUrl =
        import.meta.env.VITE_BACKEND_API_URL || "http://127.0.0.1:8000";
      const response = await axios.post(`${apiUrl}/predict/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
    } catch (err) {
      console.error("❌ Error connecting to backend:", err);
      alert("Error predicting image. Check if backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-6 rounded-2xl shadow-md w-full max-w-md text-center">
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="mb-4"
      />
      {preview && (
        <img
          src={preview}
          alt="preview"
          className="rounded-md mb-4 w-full object-cover"
        />
      )}

      <button
        onClick={handleUpload}
        className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition"
        disabled={loading}
      >
        {loading ? "Predicting..." : "Upload & Predict"}
      </button>

      {/* --- RESULT DISPLAY --- */}
      {result && (
        <div className="mt-4">
          <h2 className="text-lg font-semibold text-gray-800">Result:</h2>

          {result.class_name === "Unknown" ? (
            <p className="text-red-600 font-bold mt-2">
              ⚠️ Unknown image 
            </p>
          ) : (
            <p className="text-green-700 font-medium mt-2">
              Predicted Disease:{" "}
              <span className="font-bold">{result.class_name}</span>
            </p>
          )}

          
          <div className="w-full bg-gray-200 rounded-full h-3 mt-2">
            <div
              className={`h-3 rounded-full ${
                result.class_name === "Unknown"
                  ? "bg-red-500"
                  : "bg-green-600"
              }`}
              style={{
                width: `${(result.confidence * 100).toFixed(2)}%`,
              }}
            ></div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
