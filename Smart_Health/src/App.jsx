import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Line, Bar } from "react-chartjs-2";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const App = () => {
  // Request notification permission on load
  useEffect(() => {
    // Check if the browser supports notifications
    if ("Notification" in window) {
      // If permission hasn't been granted or denied yet
      if (
        Notification.permission !== "granted" &&
        Notification.permission !== "denied"
      ) {
        // Request permission
        Notification.requestPermission().then((permission) => {
          console.log(`Notification permission ${permission}`);
        });
      }
    }
  }, []);
  const navigate = useNavigate();
  
  // Initialize user profile from localStorage
  const userDataString = localStorage.getItem('user');
  const userData = userDataString ? JSON.parse(userDataString) : null;
  
  // State for user profile
  const [userProfile, setUserProfile] = useState({
    name: userData?.name || "John Doe",
    height: userData?.height || 175, // cm
    weight: userData?.weight || 70, // kg
    get bmi() {
      return (this.weight / Math.pow(this.height / 100, 2)).toFixed(1);
    },
  });
  
  // Function to handle logout
  const handleLogout = () => {
    localStorage.removeItem('user');
    navigate('/login');
  };

  // State for vital signs data
  const [vitalSigns, setVitalSigns] = useState({
    heartRate: 75,
    bloodPressure: { systolic: 120, diastolic: 80 },
    bodyTemperature: 36.6,
    oxygenSaturation: 98,
    activityLevel: 0,
    steps: 0,
  });

  // Function to download Excel report
  const downloadExcelReport = async () => {
    try {
      toast.info("Generating Excel report...");
      const response = await fetch("http://127.0.0.1:5000/report");
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `health_report_${new Date().toISOString().split('T')[0]}.xlsx`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast.success("Excel report downloaded successfully!");
    } catch (error) {
      console.error("Error downloading Excel report:", error);
      toast.error("Failed to download Excel report");
    }
  };

  // Function to fetch risk levels data
  const fetchRiskLevels = async () => {
    try {
      const response = await fetch("http://localhost:5000/risk_levels");
      const data = await response.json();

      if (data) {
        setRiskLevels({
          today: {
            level: data.today.risk_category.replace(" Risk", ""),
            probability: data.today.probability,
          },
          weekly: {
            level: data.week.risk_category.replace(" Risk", ""),
            probability: data.week.probability,
          },
          monthly: {
            level: data.month.risk_category.replace(" Risk", ""),
            probability: data.month.probability,
          },
        });
      }
    } catch (error) {
      console.error("Error fetching risk levels:", error);
    }
  };

  // Function to fetch weekly trends data
  const fetchWeeklyTrends = async () => {
    try {
      const response = await fetch("http://localhost:5000/weekly_trends");
      const data = await response.json();

      // Check if we have data and update state
      if (data) {
        // Handle case where only 'Today' might be available
        const heartRateData = data.heartRate || [vitalSigns.heartRate];
        const systolicData = data.systolic || [
          vitalSigns.bloodPressure.systolic,
        ];
        const diastolicData = data.diastolic || [
          vitalSigns.bloodPressure.diastolic,
        ];
        const temperatureData = data.temperature || [
          vitalSigns.bodyTemperature,
        ];
        const oxygenData = data.oxygen || [vitalSigns.oxygenSaturation];
        const stepsData = data.steps || [vitalSigns.steps];
        const datesData = data.dates || ["Today"];

        // Update historical data
        setHistoricalData({
          dates: datesData,
          heartRate: heartRateData,
          systolic: systolicData,
          diastolic: diastolicData,
          temperature: temperatureData,
          oxygen: oxygenData,
          steps: stepsData,
        });

        // Get today's data (last item in each array)
        const todayHeartRate = heartRateData[heartRateData.length - 1];
        const todaySystolic = systolicData[systolicData.length - 1];
        const todayDiastolic = diastolicData[diastolicData.length - 1];
        const todayTemperature = temperatureData[temperatureData.length - 1];
        const todayOxygen = oxygenData[oxygenData.length - 1];

        // Update baseline indices with today's data
        setBaselineIndices((prevBaseline) => ({
          heartRate: {
            ...prevBaseline.heartRate,
            user: todayHeartRate,
          },
          bloodPressure: {
            systolic: {
              ...prevBaseline.bloodPressure.systolic,
              user: todaySystolic,
            },
            diastolic: {
              ...prevBaseline.bloodPressure.diastolic,
              user: todayDiastolic,
            },
          },
          temperature: {
            ...prevBaseline.temperature,
            user: todayTemperature,
          },
          oxygen: {
            ...prevBaseline.oxygen,
            user: todayOxygen,
          },
        }));
      }
    } catch (error) {
      console.error("Error fetching weekly trends:", error);
    }
  };

  // Function to fetch latest alert
  const fetchLatestAlert = async () => {
    try {
      const response = await fetch(
        "http://127.0.0.1:5000/latest_alert?user_id=User_001"
      );
      const data = await response.json();
      const latestAlert = data.latest_alert;

      if (
        latestAlert &&
        (!lastAlertIdRef.current || latestAlert.id > lastAlertIdRef.current)
      ) {
        // Update the last seen alert ID
        lastAlertIdRef.current = latestAlert.id;

        // Check if the alert is within 1 minute
        const alertTime = new Date(latestAlert.timestamp);
        const currentTime = new Date();
        const diffInMinutes = (currentTime - alertTime) / (1000 * 60);

        if (diffInMinutes <= 1) {
          // Add to alerts list
          const newAlert = {
            id: latestAlert.id,
            message: latestAlert.alert_message,
            time: "Just now",
            severity: latestAlert.alert_type.includes("Rule")
              ? "warning"
              : "danger",
          };

          setAlerts((prevAlerts) => [newAlert, ...prevAlerts.slice(0, 9)]);

          // Show toast notification
          toast.warning(latestAlert.alert_message, {
            position: "top-right",
            autoClose: 5000,
            hideProgressBar: false,
            closeOnClick: true,
            pauseOnHover: true,
            draggable: true,
          });

          // Show browser notification if permission is granted
          if (
            "Notification" in window &&
            Notification.permission === "granted"
          ) {
            new Notification("Health Alert", {
              body: latestAlert.alert_message,
              icon: "/alert-icon.png", // You may want to add an actual icon to your public folder
            });
          }
        }
      }
    } catch (error) {
      console.error("Error fetching alert data:", error);
    }
  };

  // Function to fetch latest health data
  const fetchLatestData = async () => {
    try {
      const response = await fetch(
        "http://127.0.0.1:5000/latest?user_id=User_001"
      );
      const data = await response.json();
      const latestData = data.latest_data;

      // Convert activity_level to number and add to existing steps
      const newSteps = parseInt(latestData.activity_level) || 0;

      setVitalSigns((prevState) => ({
        heartRate: latestData.heart_rate,
        bloodPressure: {
          systolic: latestData.systolic_bp,
          diastolic: latestData.diastolic_bp,
        },
        bodyTemperature: latestData.temperature,
        oxygenSaturation: latestData.oxygen_saturation,
        activityLevel: parseFloat(latestData.activity_level),
        steps: prevState.steps + newSteps, // Add new steps to existing count
      }));
    } catch (error) {
      console.error("Error fetching health data:", error);
    }
  };

  // State for historical vital signs data (last 7 days)
  const [historicalData, setHistoricalData] = useState({
    dates: ["Today"],
    heartRate: [75],
    systolic: [120],
    diastolic: [80],
    temperature: [36.6],
    oxygen: [98],
    steps: [0],
  });

  // State for baseline health indices
  const [baselineIndices, setBaselineIndices] = useState({
    heartRate: { min: 60, max: 100, avg: 73, user: 75 },
    bloodPressure: {
      systolic: { min: 90, max: 140, avg: 120, user: 120 },
      diastolic: { min: 60, max: 90, avg: 80, user: 80 },
    },
    temperature: { min: 36.1, max: 37.2, avg: 36.6, user: 36.6 },
    oxygen: { min: 95, max: 100, avg: 97, user: 98 },
  });

  // State for risk level tracking
  const [riskLevels, setRiskLevels] = useState({
    today: { level: "Low", count: 0, probability: 0 },
    weekly: { level: "Low", count: 0, probability: 0 },
    monthly: { level: "Low", count: 0, probability: 0 },
  });

  // State for alerts
  const [alerts, setAlerts] = useState([]);

  // Reference to store the last seen alert ID
  const lastAlertIdRef = useRef(null);

  // Fetch data every 5 seconds
  useEffect(() => {
    // Initial fetch
    fetchLatestData();
    fetchLatestAlert();
    fetchWeeklyTrends();
    fetchRiskLevels();

    // Set up interval for subsequent fetches
    const dataInterval = setInterval(fetchLatestData, 2000);
    const alertInterval = setInterval(fetchLatestAlert, 3000);
    const trendsInterval = setInterval(fetchWeeklyTrends, 60000); // Update trends every minute
    const riskLevelsInterval = setInterval(fetchRiskLevels, 60000); // Update risk levels every minute

    // Cleanup interval on component unmount
    return () => {
      clearInterval(dataInterval);
      clearInterval(alertInterval);
      clearInterval(trendsInterval);
      clearInterval(riskLevelsInterval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 text-gray-800 flex flex-col">
      <ToastContainer />
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-700 to-blue-500 text-white shadow-lg">
        <div className="container mx-auto p-4 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold">
              Smart Health Monitoring and Alert System
            </h1>
            <p className="text-sm">Team: COMP4436-25-P5</p>
          </div>
          <button 
            onClick={handleLogout}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-blue-700"
          >
            Logout
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto p-4 flex-grow">
        {/* Introduction */}
        <section className="bg-white rounded-lg shadow p-4 mb-6">
          <h2 className="text-xl font-semibold mb-2">Project Overview</h2>
          <p>
            Our Smart Health Monitoring system continuously tracks vital health
            metrics, providing real-time alerts for anomalies and supporting
            long-term health trend analysis. This solution integrates wearable
            sensors with cloud-based processing to deliver comprehensive health
            insights for preventive healthcare.
          </p>
        </section>

        {/* Real-time Vital Signs */}
        <div className="flex items-center mb-4">
          <h2 className="text-2xl font-bold">Current Health Status</h2>
          <span className="ml-3 px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">
            Live Data
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow p-4 flex flex-col items-center">
            <div className="text-red-500 text-3xl mb-2">❤️</div>
            <h3 className="text-lg font-semibold">Heart Rate</h3>
            <p className="text-3xl font-bold">{vitalSigns.heartRate}</p>
            <p className="text-sm text-gray-500">BPM</p>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-red-500 h-2 rounded-full"
                style={{ width: `${(vitalSigns.heartRate - 40) / 1.4}%` }}
              ></div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4 flex flex-col items-center">
            <div className="text-blue-500 text-3xl mb-2">💉</div>
            <h3 className="text-lg font-semibold">Blood Pressure</h3>
            <p className="text-3xl font-bold">
              {vitalSigns.bloodPressure.systolic}/
              {vitalSigns.bloodPressure.diastolic}
            </p>
            <p className="text-sm text-gray-500">mmHg</p>
            <div className="mt-2 w-full flex gap-1">
              <div className="h-2 bg-blue-500 rounded-full flex-grow"></div>
              <div className="h-2 bg-red-500 rounded-full flex-grow"></div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4 flex flex-col items-center">
            <div className="text-orange-500 text-3xl mb-2">🌡️</div>
            <h3 className="text-lg font-semibold">Temperature</h3>
            <p className="text-3xl font-bold">{vitalSigns.bodyTemperature}</p>
            <p className="text-sm text-gray-500">°C</p>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${
                  parseFloat(vitalSigns.bodyTemperature) > 37.5
                    ? "bg-red-500"
                    : "bg-green-500"
                }`}
                style={{
                  width: `${
                    (parseFloat(vitalSigns.bodyTemperature) - 35) * 50
                  }%`,
                }}
              ></div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4 flex flex-col items-center">
            <div className="text-purple-500 text-3xl mb-2">🫁</div>
            <h3 className="text-lg font-semibold">Oxygen</h3>
            <p className="text-3xl font-bold">{vitalSigns.oxygenSaturation}%</p>
            <p className="text-sm text-gray-500">SpO2</p>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-purple-500 h-2 rounded-full"
                style={{ width: `${vitalSigns.oxygenSaturation}%` }}
              ></div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4 flex flex-col items-center">
            <div className="text-green-500 text-3xl mb-2">🏃</div>
            <h3 className="text-lg font-semibold">Activity</h3>
            <div className="flex items-baseline">
              <p className="text-3xl font-bold">
                {vitalSigns.steps.toLocaleString()}
              </p>
              <p className="text-sm text-gray-500 ml-1">steps</p>
            </div>
            <p className="text-sm font-medium mt-1 text-blue-600">
              {vitalSigns.activityLevel >= 0.8
                ? "High"
                : vitalSigns.activityLevel >= 0.4
                ? "Moderate"
                : "Low"}
            </p>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full"
                style={{ width: `${Math.min(vitalSigns.steps / 100, 100)}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* User Profile, Alerts and System Status */}
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          {/* User Profile Card */}
          <div className="bg-white rounded-lg shadow p-4 w-full md:w-1/3">
            <h2 className="text-lg font-semibold mb-3 flex items-center">
              <span className="text-blue-500 mr-2">👤</span>
              User Profile
            </h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="font-medium">Name</span>
                <span className="text-gray-600">{userProfile.name}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="font-medium">Height</span>
                <span className="text-gray-600">{userProfile.height} cm</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="font-medium">Weight</span>
                <span className="text-gray-600">{userProfile.weight} kg</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="font-medium">BMI</span>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${
                    userProfile.bmi < 18.5
                      ? "bg-blue-100 text-blue-800"
                      : userProfile.bmi < 25
                      ? "bg-green-100 text-green-800"
                      : userProfile.bmi < 30
                      ? "bg-yellow-100 text-yellow-800"
                      : "bg-red-100 text-red-800"
                  }`}
                >
                  {userProfile.bmi} (
                  {userProfile.bmi < 18.5
                    ? "Underweight"
                    : userProfile.bmi < 25
                    ? "Normal"
                    : userProfile.bmi < 30
                    ? "Overweight"
                    : "Obese"}
                  )
                </span>
              </div>
            </div>
          </div>
          {/* Risk Level Summary Card */}
          <div className="bg-white rounded-lg shadow p-4 w-full md:w-1/3">
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-lg font-semibold flex items-center">
                <span className="text-yellow-500 mr-2">⚠️</span>
                Risk Level Summary
              </h2>
              <button 
                onClick={downloadExcelReport}
                className="px-3 py-1 bg-green-500 text-white rounded-md hover:bg-green-600 text-sm flex items-center"
              >
                <span className="mr-1">📊</span> Export Excel
              </button>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="font-medium">Today</span>
                <div className="flex items-center">
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      riskLevels.today.level === "High"
                        ? "bg-red-100 text-red-800"
                        : riskLevels.today.level === "Medium"
                        ? "bg-yellow-100 text-yellow-800"
                        : "bg-green-100 text-green-800"
                    }`}
                  >
                    {riskLevels.today.level}
                  </span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="font-medium">This Week</span>
                <div className="flex items-center">
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      riskLevels.weekly.level === "High"
                        ? "bg-red-100 text-red-800"
                        : riskLevels.weekly.level === "Medium"
                        ? "bg-yellow-100 text-yellow-800"
                        : "bg-green-100 text-green-800"
                    }`}
                  >
                    {riskLevels.weekly.level}
                  </span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="font-medium">This Month</span>
                <div className="flex items-center">
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      riskLevels.monthly.level === "High"
                        ? "bg-red-100 text-red-800"
                        : riskLevels.monthly.level === "Medium"
                        ? "bg-yellow-100 text-yellow-800"
                        : "bg-green-100 text-green-800"
                    }`}
                  >
                    {riskLevels.monthly.level}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Alerts Card */}
          <div className="bg-white rounded-lg shadow p-4 w-full md:w-1/3">
            <h2 className="text-lg font-semibold mb-2 flex items-center">
              <span className="text-red-500 mr-2">🔔</span>
              Recent Alerts
            </h2>
            <div className="max-h-60 overflow-y-auto">
              {alerts.map((alert) => (
                <div
                  key={alert.id}
                  className={`p-2 mb-2 rounded-lg ${
                    alert.severity === "warning"
                      ? "bg-yellow-100 border-l-4 border-yellow-500"
                      : alert.severity === "danger"
                      ? "bg-red-100 border-l-4 border-red-500"
                      : alert.severity === "success"
                      ? "bg-green-100 border-l-4 border-green-500"
                      : "bg-blue-100 border-l-4 border-blue-500"
                  }`}
                >
                  <div className="flex justify-between">
                    <span className="font-medium">{alert.message}</span>
                    <span className="text-xs text-gray-500">{alert.time}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Aggregated Weekly Trends */}
        <h2 className="text-2xl font-bold mb-4">Weekly Health Trends</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Heart Rate Chart */}
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-2">Heart Rate Trend</h3>
            <div className="h-72 bg-gray-50 rounded-lg flex items-center justify-center p-4">
              <Line
                data={{
                  labels: historicalData.dates,
                  datasets: [
                    {
                      label: "Heart Rate (BPM)",
                      data: historicalData.heartRate,
                      borderColor: "rgb(239, 68, 68)",
                      backgroundColor: "rgba(239, 68, 68, 0.2)",
                      fill: true,
                      tension: 0.4,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      min: 60,
                      max: 100,
                    },
                  },
                  plugins: {
                    legend: {
                      display: true,
                    },
                    tooltip: {
                      callbacks: {
                        label: function (context) {
                          return `Heart Rate: ${context.parsed.y} BPM`;
                        },
                      },
                    },
                  },
                }}
              />
            </div>
          </div>

          {/* Blood Pressure Chart */}
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-2">Blood Pressure Trend</h3>
            <div className="h-72 bg-gray-50 rounded-lg flex items-center justify-center p-4">
              <Line
                data={{
                  labels: historicalData.dates,
                  datasets: [
                    {
                      label: "Systolic (mmHg)",
                      data: historicalData.systolic,
                      borderColor: "rgb(239, 68, 68)",
                      backgroundColor: "rgba(239, 68, 68, 0.1)",
                      tension: 0.4,
                    },
                    {
                      label: "Diastolic (mmHg)",
                      data: historicalData.diastolic,
                      borderColor: "rgb(59, 130, 246)",
                      backgroundColor: "rgba(59, 130, 246, 0.1)",
                      tension: 0.4,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      min: 40,
                      max: 160,
                    },
                  },
                }}
              />
            </div>
          </div>

          {/* Temperature Chart */}
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-2">
              Body Temperature Trend
            </h3>
            <div className="h-72 bg-gray-50 rounded-lg flex items-center justify-center p-4">
              <Line
                data={{
                  labels: historicalData.dates,
                  datasets: [
                    {
                      label: "Temperature (°C)",
                      data: historicalData.temperature,
                      borderColor: "rgb(249, 115, 22)",
                      backgroundColor: "rgba(249, 115, 22, 0.2)",
                      fill: true,
                      tension: 0.4,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      min: 35.5,
                      max: 37.5,
                    },
                  },
                }}
              />
            </div>
          </div>

          {/* Oxygen Level Chart */}
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-2">
              Oxygen Saturation Trend
            </h3>
            <div className="h-72 bg-gray-50 rounded-lg flex items-center justify-center p-4">
              <Line
                data={{
                  labels: historicalData.dates,
                  datasets: [
                    {
                      label: "SpO2 (%)",
                      data: historicalData.oxygen,
                      borderColor: "rgb(139, 92, 246)",
                      backgroundColor: "rgba(139, 92, 246, 0.2)",
                      fill: true,
                      tension: 0.4,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      min: 90,
                      max: 100,
                    },
                  },
                }}
              />
            </div>
          </div>
        </div>

        {/* Baseline Health Indices */}
        <h2 className="text-2xl font-bold mb-4">Baseline Health Indices</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Heart Rate Baseline */}
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-2">Heart Rate Baseline</h3>
            <div className="h-72 bg-gray-50 rounded-lg flex items-center justify-center p-4">
              <Bar
                data={{
                  labels: ["Minimum", "Average", "Your Baseline", "Maximum"],
                  datasets: [
                    {
                      label: "Heart Rate (BPM)",
                      data: [
                        baselineIndices.heartRate.min,
                        baselineIndices.heartRate.avg,
                        baselineIndices.heartRate.user,
                        baselineIndices.heartRate.max,
                      ],
                      backgroundColor: [
                        "rgba(59, 130, 246, 0.6)",
                        "rgba(16, 185, 129, 0.6)",
                        "rgba(245, 158, 11, 0.6)",
                        "rgba(239, 68, 68, 0.6)",
                      ],
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      min: 40,
                      max: 120,
                    },
                  },
                  indexAxis: "y",
                }}
              />
            </div>
          </div>

          {/* Blood Pressure Baseline */}
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-2">
              Blood Pressure Baseline
            </h3>
            <div className="h-72 bg-gray-50 rounded-lg flex items-center justify-center p-4">
              <Bar
                data={{
                  labels: ["Minimum", "Average", "Your Baseline", "Maximum"],
                  datasets: [
                    {
                      label: "Systolic (mmHg)",
                      data: [
                        baselineIndices.bloodPressure.systolic.min,
                        baselineIndices.bloodPressure.systolic.avg,
                        baselineIndices.bloodPressure.systolic.user,
                        baselineIndices.bloodPressure.systolic.max,
                      ],
                      backgroundColor: "rgba(239, 68, 68, 0.6)",
                    },
                    {
                      label: "Diastolic (mmHg)",
                      data: [
                        baselineIndices.bloodPressure.diastolic.min,
                        baselineIndices.bloodPressure.diastolic.avg,
                        baselineIndices.bloodPressure.diastolic.user,
                        baselineIndices.bloodPressure.diastolic.max,
                      ],
                      backgroundColor: "rgba(59, 130, 246, 0.6)",
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  indexAxis: "y",
                }}
              />
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white p-4 mt-6">
        <div className="container mx-auto text-center">
          <p>
            © 2023 COMP4436-25-P5 Team | Smart Health Monitoring and Alert
            System
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
