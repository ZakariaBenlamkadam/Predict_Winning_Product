import React from 'react';
import '../style/PredictForm.css'; 

function PredictionForm({ formData, setFormData, handleSubmit, selectedAlgorithm, handleAlgorithmChange, prediction }) {
  return (
    <div className="form-container" id="form-section">
      <h1 className="text">Product Prediction Form</h1>
      <form onSubmit={handleSubmit}>
        <div className="button-container">
          {['sport', 'phone', 'mini', 'men', 'women', 'outdoor',
            'color', 'light', 'fit', 'cloth', 'decor', 'set', 'waterproof', 
            'eye', 'new', 'pro', 'accessori', 'watch', 'cover', 'home', 'led',
            'holder', 'accessories', 'sports', 'camping', 'phones', 'health', 'technology', 
            'beauty', 'electronic'].map(element => (
            <div key={element}>
              <button
                type="button"
                className={formData[element] === 1 ? "active" : ""}
                onClick={() => setFormData((prevFormData) => ({ ...prevFormData, [element]: prevFormData[element] === 1 ? 0 : 1 }))}
              >
                {element.charAt(0).toUpperCase() + element.slice(1)}: {formData[element]}
              </button>
            </div>
          ))}
        </div>
  
        <div className='algorithm-selection'>
          <label htmlFor="algorithm">Select Algorithm:</label>
          <select id="algorithm" name="algorithm" onChange={handleAlgorithmChange} value={selectedAlgorithm}>
            <option value="logistic">Logistic Regression</option>
            <option value="rf">Random Forest</option>
          </select>
        </div>

        <button type="submit">Predict</button>
      </form>

      {prediction && <p>Prediction: {prediction}</p>}
    </div>
  );
}

export default PredictionForm;
