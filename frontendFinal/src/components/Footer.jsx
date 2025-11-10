import React from "react";

function Footer() {
    const currentYear = new Date().getFullYear();

  return (
    <div className="footer-container">
      <footer className="py-3 my-4">
        <p className="text-center text-body-secondary footer-text">© {currentYear} F1 Predictor, Inc</p>
      </footer>
    </div>
  );
}

export default Footer;