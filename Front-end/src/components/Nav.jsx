import React from 'react';
import '../style/Nav.css'; 

const Nav = () => {
  return (
    <nav className="navbar">
      <div className="left">
        {/* Logo */}
        <img
          src="./logo.png"
          alt="Your Logo"
          className="logo"
        />
      </div>
      <div className="right">
        {/* Navigation Links */}
        <ul className="nav-links">
          <li>Categories</li>
          <li>How It Works</li>
          <li>About Us</li>
        </ul>
        {/* Moon Icon */}
        <img src='./moon.png' className='moon' alt='moon'/>
      </div>
    </nav>
  );
};

export default Nav;
