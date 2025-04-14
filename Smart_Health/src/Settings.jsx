import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Error boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Settings component error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-100 flex items-center justify-center">
          <div className="bg-white p-8 rounded-lg shadow-md">
            <h2 className="text-2xl font-bold text-red-600 mb-4">Something went wrong</h2>
            <button
              onClick={() => {
                this.setState({ hasError: false });
                window.location.reload();
              }}
              className="bg-teal-600 text-white px-4 py-2 rounded hover:bg-teal-700"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

const Settings = () => {
  const navigate = useNavigate();
  const [settings, setSettings] = useState({
    apiEndpoint: 'https://api.james913.xyz/v1',
    skToken: '',
    model: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);

  // Load settings from localStorage on component mount
  useEffect(() => {
    try {
      const savedSettings = localStorage.getItem('appSettings');
      if (savedSettings) {
        const parsedSettings = JSON.parse(savedSettings);
        // Merge saved settings with default settings to ensure all properties exist
        setSettings(prevSettings => ({
          ...prevSettings,
          ...parsedSettings
        }));
      }
    } catch (error) {
      console.error('Error loading settings:', error);
      setError('Failed to load settings');
      toast.error('Failed to load settings');
    }
  }, []);

  // Save settings to localStorage
  const saveSettings = () => {
    setIsSaving(true);
    try {
      // Save to localStorage
      localStorage.setItem('appSettings', JSON.stringify(settings));
      
      // Show success message
      toast.success('Settings saved successfully!');
      
      // Log the saved settings
      console.log('Settings saved:', settings);
    } catch (error) {
      console.error('Error saving settings:', error);
      setError('Failed to save settings');
      toast.error('Failed to save settings');
    } finally {
      setIsSaving(false);
    }
  };

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    
    try {
      setSettings(prev => ({
        ...prev,
        [name]: value
      }));
    } catch (error) {
      console.error('Error handling change:', error);
      setError('Failed to update setting');
      toast.error('Failed to update setting');
    }
  };

  // Test API connection
  const testApiConnection = async () => {
    setIsLoading(true);
    setError(null);
    try {
      // Send a simple request to test the OpenAI API connection
      const response = await fetch(`${settings.apiEndpoint}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${settings.skToken}`
        },
        body: JSON.stringify({
          model: settings.model || "gpt-4o-mini",
          messages: [
            {
              role: "user",
              content: "Hello, this is a test message to verify API connectivity."
            }
          ],
          max_tokens: 10
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('API test response:', data);
      
      toast.success('OpenAI API connection successful!');
    } catch (error) {
      console.error('Error testing API connection:', error);
      setError('Failed to connect to API');
      toast.error(`Failed to connect to API: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  if (error) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="bg-white p-8 rounded-lg shadow-md">
          <h2 className="text-2xl font-bold text-red-600 mb-4">Error</h2>
          <p className="text-gray-700 mb-4">{error}</p>
          <button
            onClick={() => setError(null)}
            className="bg-teal-600 text-white px-4 py-2 rounded hover:bg-teal-700"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 text-gray-800 flex flex-col">
      {/* Header */}
      <header className="bg-gradient-to-r from-teal-600 to-teal-500 text-white shadow-lg">
        <div className="container mx-auto p-4 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold">Settings</h1>
            <p className="text-sm">Manage Your Account and Preferences</p>
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
        {/* API Configuration */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-2xl font-bold mb-4">API Configuration</h2>
          
          <div className="mb-4">
            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="apiEndpoint">
              API Endpoint
            </label>
            <div className="flex">
              <input
                type="text"
                id="apiEndpoint"
                name="apiEndpoint"
                value={settings.apiEndpoint}
                onChange={handleChange}
                className="shadow appearance-none border rounded-l py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline flex-grow"
                placeholder="https://api.openai.com/v1"
              />
              <button
                onClick={testApiConnection}
                disabled={isLoading}
                className={`bg-teal-500 hover:bg-teal-700 text-white font-bold py-2 px-4 rounded-r ${
                  isLoading ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {isLoading ? 'Testing...' : 'Test Connection'}
              </button>
            </div>
          </div>

          <div className="mb-4">
            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="skToken">
              SK Token
            </label>
            <input
              type="password"
              id="skToken"
              name="skToken"
              value={settings.skToken}
              onChange={handleChange}
              className="shadow appearance-none border rounded py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline w-full"
              placeholder="Enter your SK token"
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="model">
              Model
            </label>
            <input
              type="text"
              id="model"
              name="model"
              value={settings.model}
              onChange={handleChange}
              className="shadow appearance-none border rounded py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline w-full"
              placeholder="Enter model name (e.g. gpt-4o-mini)"
            />
          </div>
        </div>
        
        <div className="flex justify-end">
          <button
            onClick={saveSettings}
            disabled={isSaving}
            className={`bg-teal-600 hover:bg-teal-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline ${
              isSaving ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            {isSaving ? 'Saving...' : 'Save Settings'}
          </button>
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

export default Settings; 
