// src/components/BeamSearchVisualizer.js
import React, { useState, useEffect, useMemo } from 'react';
import BeamTreeVisualizer from './BeamTreeVisualizer';
import 'katex/dist/katex.min.css';
import { InlineMath } from 'react-katex';

const MathText = ({ children }) => {
  if (!children) return null;
  
  // 简单处理：如果有$符号就当作数学公式
  if (children.includes('$')) {
    // 分割文本和数学公式
    const parts = children.split(/(\$[^$]*\$)/);
    
    return (
      <span>
        {parts.map((part, index) => {
          if (part.startsWith('$') && part.endsWith('$')) {
            const math = part.slice(1, -1); // 移除$符号
            try {
              return <InlineMath key={index} math={math} />;
            } catch (error) {
              return <span key={index}>{part}</span>;
            }
          }
          return <span key={index}>{part}</span>;
        })}
      </span>
    );
  }
  
  return <span>{children}</span>;
};

const API_BASE_URL = 'https://beamsearch-demo.ngrok.io';

const BeamSearchVisualizer = () => {
  // State management
  
  const [selectedNode, setSelectedNode] = useState(null);
  const [highlightedPath, setHighlightedPath] = useState(null);
  const [collapsedNodes, setCollapsedNodes] = useState(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Data states
  const [currentData, setCurrentData] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [isComparisonMode, setIsComparisonMode] = useState(false);
  const [selectedModel, setSelectedModel] = useState('DeepSeek-R1');
  
  // Form states
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [problems, setProblems] = useState([]);
  const [selectedProblem, setSelectedProblem] = useState('');
  const [availableModels, setAvailableModels] = useState([]);

  // derive the raw problem from the list
  const rawProblem = useMemo(
    () => problems.find(p => p.id === selectedProblem) || null,
    [problems, selectedProblem]
  );
  const datasetProblems = useMemo(
    () => problems.filter(p => p.dataset === selectedDataset),
    [problems, selectedDataset]
  );
  const problemInfo = rawProblem;

  // Reset previous results when user selects a new problem
  useEffect(() => {
    setCurrentData(null);
    setComparisonData(null);
    setError(null);
    setLoading(false);
  }, [selectedProblem]);

  // API functions
  const loadDatasets = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/datasets`, {
        headers: { 'ngrok-skip-browser-warning': 'true' }
      });
      const data = await response.json();
      setDatasets(data.datasets);
      if (data.datasets.length > 0) {
        setSelectedDataset(data.datasets[0].id);
      }
    } catch (error) {
      console.error('Failed to load datasets:', error);
    }
  };

  const loadProblems = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/problems`, {
        headers: { 'ngrok-skip-browser-warning': 'true' }
      });
      const data = await response.json();
      setProblems(data.problems);
    } catch (error) {
      console.error('Failed to load problems:', error);
    }
  };

  const loadModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/models`, {
        headers: { 'ngrok-skip-browser-warning': 'true' }
      });
      const data = await response.json();
      setAvailableModels(data.models);
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };

  useEffect(() => {
    loadDatasets();
    loadProblems();
    loadModels();
  }, []);

  // Generation functions
  const generateSingleModel = async (problemId, model) => {
    setLoading(true);
    setError(null);
    setIsComparisonMode(false);
    setComparisonData(null);
    try {
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify({ problem_id: problemId, model, beam_width: 3 })
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      setCurrentData(result);
    } catch (error) {
      setError('Failed to generate reasoning: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const generateComparison = async (problemId) => {
    setLoading(true);
    setError(null);
    setIsComparisonMode(true);
    setCurrentData(null);
    try {
      const response = await fetch(`${API_BASE_URL}/compare`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify({ problem_id: problemId, beam_width: 3 })
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      setComparisonData(result);
      const firstSuccessfulModel = Object.keys(result.results).find(
        m => result.results[m].success
      );
      if (firstSuccessfulModel) setSelectedModel(firstSuccessfulModel);
    } catch (error) {
      setError('Failed to generate comparison: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Determine what to display in the tree
  const getCurrentDisplayData = () => {
    if (isComparisonMode && comparisonData) {
      const modelResult = comparisonData.results[selectedModel];
      if (modelResult?.success) {
        return {
          problem: modelResult.problem,
          beam_tree: modelResult.beam_tree,
          paths: modelResult.paths
        };
      }
    }
    return currentData;
  };
  const displayData = getCurrentDisplayData();

  const getQualityColor = score => {
    if (score >= 0.9) return '#166534';
    if (score >= 0.7) return '#10b981';
    if (score >= 0.5) return '#86efac';
    return '#EF4444';
  };

  // Rendering loading / error states
  if (loading) {
    return (
      <div className="w-full max-w-7xl mx-auto p-6 bg-gray-50 min-h-screen flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-lg p-8 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-800 mb-2">
            {isComparisonMode ? 'Comparing AI Models...' : 'Generating AI Reasoning...'}
          </h2>
          <p className="text-gray-600">Please wait while we analyze the problem</p>
        </div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="w-full max-w-7xl mx-auto p-6 bg-gray-50 min-h-screen flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-lg p-8 text-center max-w-md">
          <div className="text-red-500 text-4xl mb-4">⚠️</div>
          <h2 className="text-xl font-semibold text-gray-800 mb-2">Generation Failed</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => setError(null)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gray-50 min-h-screen">
      {/* Header Section */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-4">
          AI Beam Search Reasoning
          {isComparisonMode && <span className="text-blue-600 ml-2">- Model Comparison</span>}
        </h1>
        
        {/* Problem Selection */}
        <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-4">
          <h3 className="font-semibold text-blue-800 mb-3">Select Problem</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-blue-700 mb-2">
                Dataset ({datasets.length} available)
              </label>
              <select
                value={selectedDataset}
                onChange={(e) => {
                  const ds = e.target.value;
                  setSelectedDataset(ds);
                  setSelectedProblem('');
                  setCurrentData(null);
                  setComparisonData(null);
                  setError(null);
                }}
                className="w-full p-2 border border-blue-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              >
                {datasets.map(ds => (
                  <option key={ds.id} value={ds.id}>
                    {ds.name} ({ds.total_problems} problems)
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-blue-700 mb-2">
                Problem ({problems.filter(p => p.dataset === selectedDataset).length})
              </label>
              <select
                value={selectedProblem}
                onChange={e => setSelectedProblem(e.target.value)}
                className="w-full p-2 border border-blue-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">Choose a problem...</option>
                {problems
                  .filter(p => p.dataset === selectedDataset)
                  .map(p => (
                    <option key={p.id} value={p.id}>
                      {p.difficulty} - {p.question.replace(/\$/g, '').slice(0, 50)}…
                    </option>
                  ))}
              </select>
            </div>
          </div>

         {/* Current Problem Display */}
        {displayData?.problem && (
          <div className="bg-white p-4 rounded border border-blue-200">
            <h4 className="font-medium text-blue-800 mb-2">Current Problem:</h4>

            {/* 完整题干 */}
            <p className="text-gray-800 text-sm mb-4 whitespace-pre-wrap">
              <MathText>{displayData.problem.question}</MathText>
            </p>

          </div>
        )}


        </div>

        {/* Model Selection and Controls */}
        <div className="bg-green-50 p-4 rounded-lg mb-4 border border-green-200">
          <h4 className="font-medium text-green-800 mb-3">AI Models & Comparison</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
            {availableModels.map(model => (
              <div
                key={model.id}
                className={`p-3 rounded-lg border ${
                  model.status === 'available'
                    ? 'border-green-300 bg-green-50'
                    : 'border-gray-300 bg-gray-50'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-sm">{model.name}</div>
                    <div className="text-xs text-gray-600">{model.description}</div>
                  </div>
                  <div
                    className={`text-xs px-2 py-1 rounded ${
                      model.status === 'available'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    {model.status === 'available' ? '✓ Ready' : '✗ Unavailable'}
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => {
                if (selectedProblem) generateSingleModel(selectedProblem, 'DeepSeek-R1');
                else setError('Please select a problem first');
              }}
              disabled={loading || !selectedProblem}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400 transition-colors"
            >
              🤖 DeepSeek R1 Analysis
            </button>
            <button
              onClick={() => {
                if (selectedProblem) generateSingleModel(selectedProblem, 'O1');
                else setError('Please select a problem first');
              }}
              disabled={loading || !selectedProblem}
              className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:bg-gray-400 transition-colors"
            >
              🧠 OpenAI O1 Analysis
            </button>
            {/* <button
              onClick={() => {
                if (selectedProblem) generateComparison(selectedProblem);
                else setError('Please select a problem first');
              }}
              disabled={loading || !selectedProblem}
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-400 transition-colors"
            >
              🔬 Compare Both Models
            </button> */}
          </div>
        </div>
      </div>

      {/* Comparison Results Section */}
      {isComparisonMode && comparisonData && (
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4">🔬 Model Comparison Results</h3>
          
          {/* Comparison Analysis */}
          {comparisonData.comparison_analysis && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="bg-blue-50 p-3 rounded">
                <h4 className="font-medium text-blue-800">Success Rate</h4>
                <p className="text-blue-700">
                  {comparisonData.comparison_analysis.successful_models}/
                  {comparisonData.comparison_analysis.total_models} models successful
                </p>
              </div>
              
              <div className="bg-green-50 p-3 rounded">
                <h4 className="font-medium text-green-800">Answer Agreement</h4>
                <p className="text-green-700">
                  {comparisonData.comparison_analysis.consensus_analysis?.agreement_percentage.toFixed(1)}% consensus
                </p>
              </div>
              
              <div className="bg-purple-50 p-3 rounded">
                <h4 className="font-medium text-purple-800">Most Common Answer</h4>
                <p className="text-purple-700">
                  {comparisonData.comparison_analysis.consensus_analysis?.most_common_answer || 'N/A'}
                </p>
              </div>
            </div>
          )}
          
          {/* Model Switching Buttons */}
          <div className="flex gap-2 mb-4">
            <span className="text-sm font-medium text-gray-700 self-center mr-2">View Model:</span>
            {Object.keys(comparisonData.results).map(modelName => {
              const result = comparisonData.results[modelName];
              return (
                <button
                  key={modelName}
                  onClick={() => setSelectedModel(modelName)}
                  className={`px-3 py-1 rounded text-sm transition-colors ${
                    selectedModel === modelName 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  } ${!result.success ? 'opacity-50' : ''}`}
                  disabled={!result.success}
                >
                  {modelName} {result.success ? '✓' : '✗'}
                </button>
              );
            })}
          </div>
          
          {/* Current Model Info */}
          {comparisonData.results[selectedModel] && (
            <div className="bg-gray-50 p-3 rounded">
              <h4 className="font-medium mb-2">Current: {selectedModel}</h4>
              <div className="text-sm text-gray-600 grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <span className="font-medium">Status:</span> 
                  {comparisonData.results[selectedModel].success ? ' ✅ Success' : ' ❌ Failed'}
                </div>
                <div>
                  <span className="font-medium">Type:</span> 
                  {comparisonData.results[selectedModel].model_info?.type || 'Unknown'}
                </div>
                <div>
                  <span className="font-medium">Paths:</span> 
                  {comparisonData.results[selectedModel].paths?.length || 0}
                </div>
                <div>
                  <span className="font-medium">Nodes:</span> 
                  {Object.keys(comparisonData.results[selectedModel].beam_tree || {}).length}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Main Content Area */}
      {(displayData || comparisonData) && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Tree Visualization */}
          <div className="lg:col-span-3">
            <BeamTreeVisualizer
              displayData={displayData}
              collapsedNodes={collapsedNodes}
              setCollapsedNodes={setCollapsedNodes}
              setSelectedNode={setSelectedNode}
              setHighlightedPath={setHighlightedPath}
            />
          </div>

          {/* Right Control Panel */}
          <div className="bg-white rounded-lg shadow-lg p-4">
            <h3 className="text-lg font-semibold mb-4">Reasoning Paths</h3>
            
            <div className="space-y-3 mb-6">
              <button
                onClick={() => window.resetHighlight()}
                className="w-full px-3 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 text-sm"
              >
                Reset View
              </button>
              
              {displayData?.paths?.map((path, idx) => (
                <div key={path.id} className="space-y-2">
                  <button
                    onClick={() => window.highlightPath(path.id)}
                    className={`w-full p-3 rounded-lg border-2 text-left transition-all ${
                      highlightedPath === path.id 
                        ? 'border-blue-500 bg-blue-50' 
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-sm">Path {idx + 1}</span>
                      <div className="flex items-center gap-2">
                        <div 
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: getQualityColor(path.score) }}
                        ></div>
                        <span className="text-xs">{path.score.toFixed(2)}</span>
                      </div>
                    </div>
                    <div className="text-xs text-gray-600 mb-1">
                      {path.nodes.length} steps
                    </div>
                    <div className="text-xs">
                      Answer: <span className="font-medium">{path.final_answer}</span>
                      {path.is_correct ? (
                        <span className="text-green-600 ml-1">✓</span>
                      ) : (
                        <span className="text-red-600 ml-1">✗</span>
                      )}
                    </div>
                  </button>
                </div>
              ))}
            </div>

            {/* Selected Node Details */}
            {selectedNode && (
              <div className="border-t pt-4">
                <h4 className="font-semibold mb-3">Node Details</h4>
                <div className="bg-gray-50 rounded-lg p-3 mb-3">
                  <div className="text-sm font-medium mb-2">{selectedNode.content}</div>
                  <div className="space-y-1">
                    {selectedNode.variables?.map((variable, idx) => (
                      <div key={idx} className="text-xs font-mono bg-white px-2 py-1 rounded border">
                        {variable}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span>Quality Score:</span>
                    <span className="font-medium">{selectedNode.quality_score?.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Probability:</span>
                    <span className="font-medium">{selectedNode.probability?.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Type:</span>
                    <span className="font-medium">{selectedNode.reasoning_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Depth:</span>
                    <span className="font-medium">{selectedNode.depth}</span>
                  </div>
                  {isComparisonMode && (
                    <div className="flex justify-between">
                      <span>Model:</span>
                      <span className="font-medium">{selectedModel}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            {/* Comparison Mode Performance Stats */}
            {isComparisonMode && comparisonData?.comparison_analysis?.model_performance && (
              <div className="border-t pt-4">
                <h4 className="font-semibold mb-3">Model Performance</h4>
                <div className="space-y-3">
                  {Object.entries(comparisonData.comparison_analysis.model_performance).map(([model, perf]) => (
                    <div key={model} className={`p-2 rounded text-xs ${
                      selectedModel === model ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50'
                    }`}>
                      <div className="font-medium mb-1">{model}</div>
                      <div className="space-y-1">
                        <div className="flex justify-between">
                          <span>Best Score:</span>
                          <span>{perf.best_score?.toFixed(3)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Avg Score:</span>
                          <span>{perf.average_score?.toFixed(3)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Answer:</span>
                          <span className="font-medium">{perf.best_answer}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Type:</span>
                          <span className="text-xs bg-gray-200 px-1 rounded">{perf.reasoning_type}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Empty State */}
      {!displayData && !comparisonData && !loading && !error && (
        <div className="bg-white rounded-lg shadow-lg p-8 text-center">
          <div className="text-6xl mb-4">🤖</div>
          <h2 className="text-xl font-semibold text-gray-800 mb-2">Ready for AI Reasoning Analysis</h2>
          <p className="text-gray-600 mb-4">
            Select a problem from the dataset above and choose an analysis mode to get started.
          </p>
          
        </div>
      )}
    </div>
  );
};

export default BeamSearchVisualizer;