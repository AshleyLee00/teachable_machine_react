import React from 'react';
import ErrorBoundary from './components/ErrorBoundary';
import TeachableMachine from './components/TeachableMachine';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <h1 className="text-3xl font-bold text-center mb-8">
        Image Classifier
      </h1>
      <ErrorBoundary>
        <TeachableMachine />
      </ErrorBoundary>
    </div>
  );
}

export default App;