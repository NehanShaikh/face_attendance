// src/config.js
const config = {
  API_URL: process.env.NODE_ENV === 'development' 
    ? 'http://localhost:3000'
    : 'https://face-attendance-9vis.onrender.com'
};
