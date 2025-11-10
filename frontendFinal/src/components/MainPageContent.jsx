import React, { useRef, useState, useEffect } from "react";
import AccuracyPesentationContainer from "./AccuracyPesentationContainer";
import RaceTable from "./RaceTable";

function MainPageContent() {
    const [showRaceTable, setShowRaceTable] = useState(false);
    const [showAccuracy, setShowAccuracy] = useState(true);

    useEffect(() => {
        const handleScroll = () => {
            // Show RaceTable when user scrolls down
            if (window.scrollY > 100) {
                setShowRaceTable(true);
                setShowAccuracy(false);
            } else {
                setShowRaceTable(false);
                setShowAccuracy(true);
            }
        };

        window.addEventListener('scroll', handleScroll);

        // Cleanup, prevents memory leaks
        return () => {
            window.removeEventListener('scroll', handleScroll);
        };
    }, []);

    // While RaceTable is visible: block downward scrolling but allow upward scrolling
    // so the user can scroll back up to return to the accuracy view.
    useEffect(() => {
        if (!showRaceTable) return;

        let startY = 0;

        const onWheel = (e) => {
            // deltaY > 0 means user is scrolling down; block that
            if (e.deltaY > 0) {
                e.preventDefault();
            } else if (e.deltaY < 0) {
                // scrolling up -> dismiss race table
                setShowRaceTable(false);
                setShowAccuracy(true);
            }
        };


        //for mobile
        const onTouchStart = (e) => {
            startY = e.touches[0].clientY;
        };

        const onTouchMove = (e) => {
            const currentY = e.touches[0].clientY;
            const dy = currentY - startY;
            // dy < 0: finger moved up => user wants to scroll down -> block
            if (dy < -10) {
                e.preventDefault();
            } else if (dy > 10) {
                // finger moved down => user wants to scroll up -> dismiss
                setShowRaceTable(false);
                setShowAccuracy(true);
            }
        };

        window.addEventListener('wheel', onWheel, { passive: false });
        window.addEventListener('touchstart', onTouchStart, { passive: true });
        window.addEventListener('touchmove', onTouchMove, { passive: false });

        return () => {
            window.removeEventListener('wheel', onWheel);
            window.removeEventListener('touchstart', onTouchStart);
            window.removeEventListener('touchmove', onTouchMove);
        };
    }, [showRaceTable]);

    return (
        <div className="main-page-content">
            <div className={`fade-section ${showAccuracy ? 'fade-in' : 'fade-out'}`}>
                <AccuracyPesentationContainer/>
            </div>
            <div className={`fade-section ${showRaceTable ? 'fade-in' : 'fade-out'}`}>
                <RaceTable />
            </div>
        </div>
    );
}

export default MainPageContent;