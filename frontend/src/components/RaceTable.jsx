import React, { useState } from "react";

function RaceTable() {
    const [raceName, setRaceName] = useState("");
    const [openTable, setOpenTable] = useState(false);

    const handleSubmit = (event) => {
        event.preventDefault();
        setOpenTable(true);
        // console.log("Race to predict:", raceName);
    };

    const handleInputChange = (event) => {
        setRaceName(event.target.value);
    };

    return (
        <div id="race-table-div">
            {/* form panel (kept in DOM so CSS transitions can animate) */}
            <div className={`rt-fade ${!openTable ? 'show' : ''}`} aria-hidden={openTable}>
                    <div className="input">
                        <h2>Which race do you want to predict?</h2>
                        <form onSubmit={handleSubmit}>
                            <div id="input-race-table">
                                <input
                                    type="text"
                                    value={raceName}
                                    onChange={handleInputChange}
                                    placeholder="Enter race name"
                                    required
                                />
                                <button type="submit">Predict Race</button>
                            </div>
                        </form>
                    </div>
                </div>
            {/* results panel (kept in DOM) */}
            <div className={`rt-fade ${openTable ? 'show' : ''}`} aria-hidden={!openTable}>
                <div id="race-results" className="race-results">
                    <h2>Predictions for the race in {raceName}</h2>
                    <table className="results-table">
                        <thead>
                            <tr>
                                <th>Driver</th>
                                <th>Team</th>
                                <th>Predicted Position</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Max Verstappen</td>
                                <td>Red Bull</td>
                                <td>1</td>
                            </tr>
                            <tr>
                                <td>Lewis Hamilton</td>
                                <td>Mercedes</td>
                                <td>2</td>
                            </tr>
                            <tr>
                                <td>Charles Leclerc</td>
                                <td>Ferrari</td>
                                <td>3</td>
                            </tr>
                        </tbody>
                        
                    </table>
                    <div style={{ display: "flex", justifyContent: "center" }}>
                        <button 
                            className="back-button" 
                            onClick={() => 
                                    {
                                        setOpenTable(false)
                                        setRaceName("")
                                    }
                                }>
                            Back
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default RaceTable;