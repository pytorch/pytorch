import React, { useState, useEffect } from 'react';
import EventGrid from './components/EventGrid';
import './App.css';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('./trace.json')
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to load trace.json');
        }
        return response.json();
      })
      .then(jsonData => {
        setData(jsonData);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="app-loading">Loading trace data...</div>;
  }

  if (error) {
    return <div className="app-error">Error: {error}</div>;
  }

  return (
    <div className="App">
      <header className="app-header">
        <h1>Distributed System Event Visualizer</h1>
      </header>
      <EventGrid data={data} />
    </div>
  );
}

export default App;
