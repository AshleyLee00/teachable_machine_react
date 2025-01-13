import React from 'react';
import ErrorBoundary from './components/ErrorBoundary';
import TeachableMachine from './components/TeachableMachine';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <ErrorBoundary>
        <TeachableMachine />
      </ErrorBoundary>
    </div>
  );
}

export default App;