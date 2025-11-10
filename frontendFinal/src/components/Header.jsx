import React from "react";

function Header() {
  return (
    <div className="header-container">
      <header className="d-flex flex-wrap align-items-center justify-content-center justify-content-md-between py-3 mb-4 border-bottom">
        <div className="col-md-3 mb-2 mb-md-0">
          <a
            href="/"
            className="d-inline-flex link-body-emphasis text-decoration-none"
          >
            {/* Replace SVG with text or simple logo */}
            <span className="fs-4 fw-bold">🏎️ F1 Predictor</span>
          </a>
        </div>
        
        <ul className="nav col-12 col-md-auto mb-2 justify-content-center mb-md-0 middle-buttons">
          <li>
            <a href="#" className="nav-link px-2 link-secondary">
              Home
            </a>
          </li>
          <li>
            <a href="#" className="nav-link px-2">
              Predictions
            </a>
          </li>
          <li>
            <a href="#" className="nav-link px-2">
              Races
            </a>
          </li>
          <li>
            <a href="#" className="nav-link px-2">
              Drivers
            </a>
          </li>
          <li>
            <a href="#" className="nav-link px-2">
              About
            </a>
          </li>
        </ul>
        
        <div className="col-md-3 text-end">
          <button type="button" className="btn btn-outline-primary me-2 login-button">
            Login
          </button>
          <button type="button" className="btn btn-primary signup-button">
            Sign-up
          </button>
        </div>
      </header>
    </div>
  );
}

export default Header;