import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const AIHealthAdvisor = () => {
  const navigate = useNavigate();
  const [healthData, setHealthData] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [riskLevels, setRiskLevels] = useState({
    today: { level: 'Low', probability: 0.1 },
    weekly: { level: 'Low', probability: 0.1 },
    monthly: { level: 'Low', probability: 0.1 }
  });

  // Load settings from localStorage
  const [settings, setSettings] = useState({
    apiEndpoint: 'http://localhost:5000'
  });

  useEffect(() => {
    const savedSettings = localStorage.getItem('appSettings');
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }
  }, []);

  // Fetch health data from the backend
  const fetchHealthData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      // Fetch current health data
      const currentResponse = await fetch(`${settings.apiEndpoint}/latest?user_id=User_001`);
      if (!currentResponse.ok) {
        throw new Error(`HTTP error! status: ${currentResponse.status}`);
      }
      const currentData = await currentResponse.json();
      const latestData = currentData.latest_data;
      
      // Fetch historical data (last 7 days)
      const historicalResponse = await fetch(`${settings.apiEndpoint}/weekly_trends?user_id=User_001`);
      if (!historicalResponse.ok) {
        throw new Error(`HTTP error! status: ${historicalResponse.status}`);
      }
      const historicalData = await historicalResponse.json();
      
      // Fetch risk levels
      const riskResponse = await fetch(`${settings.apiEndpoint}/risk_levels?user_id=User_001`);
      if (!riskResponse.ok) {
        throw new Error(`HTTP error! status: ${riskResponse.status}`);
      }
      const riskData = await riskResponse.json();
      
      // Format the data for analysis
      const formattedData = {
        user_data: {
          age: 30, // Default values, should be fetched from user profile
          gender: 'male',
          medical_history: [],
          medications: []
        },
        current_health: {
          heart_rate: latestData.heart_rate,
          blood_pressure: {
            systolic: latestData.systolic_bp,
            diastolic: latestData.diastolic_bp
          },
          temperature: latestData.temperature,
          oxygen_saturation: latestData.oxygen_saturation
        },
        historical_data: historicalData,
        risk_levels: {
          today: riskData.today.risk_category,
          weekly: riskData.week.risk_category,
          monthly: riskData.month.risk_category
        }
      };
      
      setHealthData(formattedData);
      setHistoricalData(historicalData);
      setRiskLevels({
        today: riskData.today,
        weekly: riskData.week,
        monthly: riskData.month
      });
    } catch (error) {
      console.error('Error fetching health data:', error);
      setError('Failed to fetch health data. Please try again later.');
      toast.error('Failed to fetch health data');
    } finally {
      setIsLoading(false);
    }
  };

  // Analyze health data using AI
  const analyzeHealthData = async () => {
    if (!healthData) {
      toast.error('Please fetch health data first');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      // Get settings from localStorage
      const savedSettings = JSON.parse(localStorage.getItem('appSettings') || '{}');
      
      // Call our local API endpoint for AI analysis
      const apiEndpoint = 'http://localhost:5000';
      console.log('Sending analysis request to:', `${apiEndpoint}/ai_analysis`);

      const response = await fetch(`${apiEndpoint}/ai_analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_data: healthData.user_data,
          current_health: healthData.current_health,
          historical_data: healthData.historical_data,
          risk_levels: healthData.risk_levels,
          // Pass OpenAI settings so our backend can use them
          openai_settings: {
            api_endpoint: savedSettings.apiEndpoint || 'https://api.openai.com/v1',
            api_key: savedSettings.skToken || '',
            model: savedSettings.model || 'gpt-4o-mini'
          }
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Analysis response:', data);
      
      setAnalysis(data);
      toast.success('Health analysis completed successfully!');
    } catch (error) {
      console.error('Error analyzing health data:', error);
      setError(error.message);
      toast.error('Failed to analyze health data: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Load health data on component mount
  useEffect(() => {
    fetchHealthData();
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 text-gray-800 flex flex-col">
      {/* Header */}
      <header className="bg-gradient-to-r from-teal-600 to-teal-500 text-white shadow-lg">
        <div className="container mx-auto p-4 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold">AI Health Advisor</h1>
            <p className="text-sm">Personalized Health Insights</p>
          </div>
          <div className="flex items-center space-x-4">
            <button 
              onClick={() => navigate('/')}
              className="px-4 py-2 bg-white text-teal-700 rounded-md hover:bg-teal-50 focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-teal-600"
            >
              Back to Dashboard
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto p-4 flex-grow">
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
            <strong className="font-bold">Error!</strong>
            <span className="block sm:inline"> {error}</span>
          </div>
        )}

        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-2xl font-bold mb-4">Current Health Data</h2>
          {isLoading ? (
            <div className="flex justify-center items-center h-32">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-teal-500"></div>
            </div>
          ) : healthData ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-700">Heart Rate</h3>
                <p className="text-gray-600">{healthData.current_health.heart_rate} BPM</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-700">Blood Pressure</h3>
                <p className="text-gray-600">
                  {healthData.current_health.blood_pressure.systolic}/{healthData.current_health.blood_pressure.diastolic} mmHg
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-700">Temperature</h3>
                <p className="text-gray-600">{healthData.current_health.temperature} °C</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-700">Oxygen Saturation</h3>
                <p className="text-gray-600">{healthData.current_health.oxygen_saturation}%</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-700">Risk Level (Today)</h3>
                <p className="text-gray-600">{healthData.risk_levels.today}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-700">Risk Level (Weekly)</h3>
                <p className="text-gray-600">{healthData.risk_levels.weekly}</p>
              </div>
            </div>
          ) : (
            <p className="text-gray-600">No health data available</p>
          )}
          <div className="mt-4 flex justify-end">
            <button
              onClick={fetchHealthData}
              disabled={isLoading}
              className={`bg-teal-500 hover:bg-teal-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline ${
                isLoading ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {isLoading ? 'Loading...' : 'Refresh Data'}
            </button>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-2xl font-bold mb-4">AI Health Analysis</h2>
          {isLoading ? (
            <div className="flex justify-center items-center h-32">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-teal-500"></div>
            </div>
          ) : analysis ? (
            <div className="prose max-w-none">
              <div className="space-y-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-xl font-semibold mb-2">Overall Assessment</h3>
                  <p className="whitespace-pre-wrap">{analysis.overallAssessment}</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-xl font-semibold mb-2">Metrics Analysis</h3>
                  <p className="whitespace-pre-wrap">{analysis.metricsAnalysis}</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-xl font-semibold mb-2">Risk Assessment</h3>
                  <p className="whitespace-pre-wrap">{analysis.riskAssessment}</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-xl font-semibold mb-2">Recommendations</h3>
                  <p className="whitespace-pre-wrap">{analysis.recommendations}</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-xl font-semibold mb-2">Lifestyle Suggestions</h3>
                  <p className="whitespace-pre-wrap">{analysis.lifestyleSuggestions}</p>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-gray-600">No analysis available</p>
          )}
          <div className="mt-4 flex justify-end">
            <button
              onClick={analyzeHealthData}
              disabled={isLoading || !healthData}
              className={`bg-teal-500 hover:bg-teal-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline ${
                (isLoading || !healthData) ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {isLoading ? 'Analyzing...' : 'Analyze Health Data'}
            </button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white p-4 mt-6">
        <div className="container mx-auto text-center">
          <p>© 2025 COMP4436-25-P5 Team | Smart Health Monitoring and Alert System</p>
        </div>
      </footer>
    </div>
  );
};

export default AIHealthAdvisor; 