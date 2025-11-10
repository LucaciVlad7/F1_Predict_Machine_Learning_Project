import React, { useState } from "react";

function RaceTable() {
    const [raceName, setRaceName] = useState("");
    const [openTable, setOpenTable] = useState(false);
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [modelAccuracy, setModelAccuracy] = useState(null);
    const [weatherConditions, setWeatherConditions] = useState(null);

    const handleSubmit = async (event) => {
        event.preventDefault();
        setLoading(true);
        setError(null);
        
        try {
            console.log(`Predicting race: ${raceName}`);
            
            const response = await fetch('http://localhost:5000/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    raceName: raceName,
                    year: 2025
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                setPredictions(data.predictions);
                setModelAccuracy(data.modelAccuracy);
                setWeatherConditions(data.weatherConditions);
                setOpenTable(true);
                console.log("Predictions received:", data);
            } else {
                setError(data.error || "Failed to get predictions");
            }
        } catch (err) {
            console.error("Error fetching predictions:", err);
            setError("Failed to connect to prediction service. Make sure the Flask server is running on port 5000.");
        } finally {
            setLoading(false);
        }
        
    };

    const handleInputChange = (event) => {
        setRaceName(event.target.value);
    };

    const handleBack = () => {
        setOpenTable(false);
        setRaceName("");
        setPredictions([]);
        setError(null);
        setModelAccuracy(null);
        setWeatherConditions(null);
    };

    return (
        <div id="race-table-div">
            {/* form panel (kept in DOM so CSS transitions can animate) */}
            <div className={`rt-fade ${!openTable ? 'show' : ''}`} aria-hidden={openTable}>
                    <div className="input">
                        <h2>Which race do you want to predict?</h2>
                        {error && (
                        <div style={{ 
                            color: 'red', 
                            marginBottom: '20px', 
                            textAlign: 'center',
                            backgroundColor: 'rgba(255, 0, 0, 0.1)',
                            padding: '10px',
                            borderRadius: '5px'
                        }}>
                            {error}
                        </div>
                        )}
                        <form onSubmit={handleSubmit}>
                            <div id="input-race-table">
                                <input
                                    type="text"
                                    value={raceName}
                                    onChange={handleInputChange}
                                    placeholder="Enter race name"
                                    required
                                    disabled={loading}
                                />
                                 <button type="submit" disabled={loading}>
                                {loading ? "Predicting..." : "Predict Race"}
                            </button>
                            </div>
                        </form>
                        {loading && (
                        <div style={{ textAlign: 'center', marginTop: '20px' }}>
                            <p>🤖 AI is analyzing race data...</p>
                        </div>
                        )}
                    </div>
                </div>
            {/* results panel (kept in DOM) */}
            <div className={`rt-fade ${openTable ? 'show' : ''}`} aria-hidden={!openTable}>
                <div id="race-results" className="race-results">
                    <h2>Predictions for {raceName}</h2>
                    {modelAccuracy && (
                        <p style={{ textAlign: 'center', fontSize: '1.2rem', marginBottom: '20px' }}>
                            Model Accuracy: <strong>{modelAccuracy}%</strong>
                        </p>
                    )}
                    {weatherConditions && (
                        <p style={{ textAlign: 'center', fontSize: '1rem', marginBottom: '20px' }}>
                            Weather: {weatherConditions.temperature}°C, Rain: {weatherConditions.rainProbability}%
                        </p>
                    )}
                    <table className="results-table">
                        <thead>
                            <tr>
                                <th>Position</th>
                                <th>Driver</th>
                                <th>Team</th>
                                <th>Predicted Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {predictions.length > 0 ? (
                                predictions.map((prediction, index) => (
                                    <tr key={index}>
                                        <td>{prediction.position}</td>
                                        <td>{prediction.driver}</td>
                                        <td>{prediction.team}</td>
                                        <td>{prediction.predictedTime}</td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td>Max Verstappen</td>
                                    <td>Red Bull</td>
                                    <td>1</td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                    <div style={{ display: "flex", justifyContent: "center" }}>
                        <button 
                            className="back-button" 
                            onClick={handleBack}>
                            Back
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default RaceTable;