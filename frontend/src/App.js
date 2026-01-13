// This is the main component of the frontend. It contains the form to upload the image and the logic to send the image to the backend for prediction.

import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [prediction, setPrediction] = useState('');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
      setSelectedFile(file);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      alert('Please select a file first!');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      setPrediction(`Result: The food is ${result.status}.`);
    } catch (error) {
      console.error('Error submitting the form:', error);
      alert('Failed to get the prediction. Please try again.');
    }
  };

  return (
  <div className="App">
    <header className="App-header">
      <h1>Food Quality Detector</h1>
      <div className="upload-section">
      <h3>Upload Your Food Image Here</h3>
      <form onSubmit={handleSubmit}>
        {/* Hide the actual file input and use a label as the "button" */}
        <input type="file" id="fileInput" onChange={handleFileChange} style={{ display: 'none' }} />
        <label htmlFor="fileInput" className="fileInputLabel">Choose File</label>
        <button type="submit">Submit</button>
      </form>
      {imagePreviewUrl && <img src={imagePreviewUrl} alt="Food preview" />}
      {prediction && <p>{prediction}</p>}
    </div>
    </header>
  </div>
  );
}

export default App;