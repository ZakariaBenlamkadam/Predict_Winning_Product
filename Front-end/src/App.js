import React, { useState } from 'react';
import './App.css';
import Nav from './components/Nav';
import HeroSection from './components/Hero';
import PredictionForm from './components/PredictionForm';


function App() {
  const [formData, setFormData] = useState({
    sport: '',
    phone: '',
    mini: '',
    pc: '',
    men: '',
    women: '',
    comput: '',
    outdoor: '',
    color: '',
    light: '',
    fit: '',
    cloth: '',
    decor: '',
    set: '',
    waterproof: '',
    eye: '',
    new: '',
    pro: '',
    accessori: '',
    watch: '',
    cover: '',
    home: '',
    led: '',
    holder: '',
    accessories: '',
    sports: '',
    camping: '',
    phones: '',
    health: '',
    technology: '',
    beauty: '',
    electronic: '',
  });

  const [prediction, setPrediction] = useState(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('logistic'); 

  

  const handleAlgorithmChange = (e) => {
    setSelectedAlgorithm(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const url = `http://localhost:5000/predict_${selectedAlgorithm}`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        const data = await response.json();
        setPrediction(data[`prediction_${selectedAlgorithm}`]);
      } else {
        console.error('Failed to make prediction');
      }
    } catch (error) {
      console.error('Error during prediction:', error);
    }
  };

  return (
    <div className="App">
    <Nav />
    <HeroSection/>
    
    <PredictionForm
        formData={formData}
        setFormData={setFormData}
        handleSubmit={handleSubmit}
        selectedAlgorithm={selectedAlgorithm}
        handleAlgorithmChange={handleAlgorithmChange}
        prediction={prediction}
    />
    
      
    </div>
  );
}

export default App;

