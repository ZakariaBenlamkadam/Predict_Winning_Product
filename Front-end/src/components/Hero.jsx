import React from 'react';

function HeroSection() {
  return (
    <div className="homepage-container">
      <div className="background-image">
        <div
          style={{
            position: 'absolute',
            top: '0',
            left: '0',
            width: '100%',
            height: '100%',
            background: 'url("./images/background.png")',
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            opacity: '0.8',
          }}
        ></div>
      </div>
      <div className="content">
        <h1>
          <span style={{ fontFamily: 'rockwell Extrabold', fontSize: '80px', fontWeight: 'bold', color: '#C8F560' }}>
            Predict,
          </span>
          {' '}
          <span style={{ fontFamily: 'Forte', fontSize: '80px', fontStyle: 'regular', color: '#FFFFFF' }}>
            Win,
          </span>
          {' '}
          <span style={{ fontFamily: 'Molot', fontSize: '80px', fontStyle: 'regular', color: '#A99BFC' }}>
            Succeed
          </span>
        </h1>
        <p className="descriptive-text">Elevate your product strategy with the power of predictive analytics.</p>
        <button className="button-go-to-form" onClick={() => document.getElementById('form-section').scrollIntoView({ behavior: 'smooth' })}>
          Ready? Let's Begin
        </button>
      </div>
    </div>
  );
}

export default HeroSection;
