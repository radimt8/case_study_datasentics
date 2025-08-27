// API Configuration - use /api for nginx proxy or direct for development
const API_BASE_URL = window.location.hostname === 'localhost' && window.location.port === '3000' 
    ? 'http://localhost:8000'  // Direct API for development
    : '/api';  // Proxied through nginx in production

// DOM Elements
const userIdInput = document.getElementById('userId');
const topKSlider = document.getElementById('topK');
const topKValue = document.getElementById('topKValue');
const getRecommendationsBtn = document.getElementById('getRecommendations');
const randomUserBtn = document.getElementById('randomUser');
const loadingDiv = document.getElementById('loading');
const noticeDiv = document.getElementById('notice');
const resultsDiv = document.getElementById('results');
const systemInfoDiv = document.getElementById('systemInfo');
const modelInfoDiv = document.getElementById('modelInfo');

// State
let availableUsers = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
});

function setupEventListeners() {
    // Update slider value display
    topKSlider.addEventListener('input', () => {
        topKValue.textContent = topKSlider.value;
    });

    // Get recommendations
    getRecommendationsBtn.addEventListener('click', getRecommendations);
    
    // Random user
    randomUserBtn.addEventListener('click', selectRandomUser);
    
    // Enter key on user input
    userIdInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            getRecommendations();
        }
    });
}

async function initializeApp() {
    try {
        // Load system information
        await loadSystemInfo();
        
        // Load model information
        await loadModelInfo();
        
        // Load available users
        await loadAvailableUsers();
        
    } catch (error) {
        showNotice('Failed to initialize application. Make sure the API server is running.', 'error');
        console.error('Initialization error:', error);
    }
}

async function loadSystemInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const stats = await response.json();
        
        systemInfoDiv.innerHTML = `
            <div class="stats-grid">
                <div class="stat">
                    <span class="stat-label">Users with Recommendations:</span>
                    <span class="stat-value">${stats.total_users_with_recs}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Memory Usage:</span>
                    <span class="stat-value">${stats.memory_usage_mb} MB</span>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Failed to load system info:', error);
    }
}

async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model/info`);
        if (response.ok) {
            const modelInfo = await response.json();
            
            modelInfoDiv.innerHTML = `
                <div class="model-stats-grid">
                    <div class="stat">
                        <span class="stat-label">Last Updated:</span>
                        <span class="stat-value">${new Date(modelInfo.last_updated).toLocaleString()}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Test RMSE:</span>
                        <span class="stat-value">${modelInfo.model_performance.rmse_original_scale}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Dataset Sparsity:</span>
                        <span class="stat-value">${(modelInfo.dataset_info.sparsity * 100).toFixed(1)}%</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Training Epochs:</span>
                        <span class="stat-value">${modelInfo.training_info.converged_epoch}</span>
                    </div>
                </div>
            `;
        } else {
            modelInfoDiv.innerHTML = '<p class="warning">Model not trained yet. Run batch processing first.</p>';
        }
    } catch (error) {
        console.error('Failed to load model info:', error);
        modelInfoDiv.innerHTML = '<p class="error">Failed to load model information</p>';
    }
}

async function loadAvailableUsers() {
    try {
        const response = await fetch(`${API_BASE_URL}/users`);
        const data = await response.json();
        availableUsers = data.user_ids || [];
        
        // Set a default user if available
        if (availableUsers.length > 0 && !userIdInput.value) {
            userIdInput.value = availableUsers[0];
        }
        
    } catch (error) {
        console.error('Failed to load users:', error);
    }
}

function selectRandomUser() {
    if (availableUsers.length > 0) {
        const randomIndex = Math.floor(Math.random() * availableUsers.length);
        userIdInput.value = availableUsers[randomIndex];
    }
}

async function getRecommendations() {
    const userId = userIdInput.value.trim();
    const topK = topKSlider.value;
    
    if (!userId) {
        showNotice('Please enter a user ID', 'warning');
        return;
    }
    
    showLoading(true);
    hideNotice();
    hideResults();
    
    try {
        const response = await fetch(`${API_BASE_URL}/recommend/${userId}?top_k=${topK}`);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get recommendations');
        }
        
        const data = await response.json();
        displayRecommendations(data);
        
        // Show notice if user requested more than available
        if (data.note) {
            showNotice(data.note, 'info');
        }
        
    } catch (error) {
        showNotice(`Error: ${error.message}`, 'error');
        console.error('Recommendation error:', error);
    } finally {
        showLoading(false);
    }
}

function displayRecommendations(data) {
    const recommendations = data.recommendations;
    
    if (recommendations.length === 0) {
        showNotice('No recommendations available for this user', 'info');
        return;
    }
    
    const resultsHTML = `
        <div class="results-header">
            <h2>Recommendations for User ${data.user_id}</h2>
            <p>Showing ${recommendations.length} of ${data.total_available} available recommendations</p>
        </div>
        <div class="recommendations-grid">
            ${recommendations.map((rec, index) => `
                <div class="recommendation-card">
                    <div class="rank">#${index + 1}</div>
                    <div class="book-info">
                        <h3 class="book-title">${rec.title}</h3>
                        <div class="rating-info">
                            <div class="predicted-rating">
                                <span class="rating-value">${rec.predicted_rating}/10</span>
                                <span class="rating-label">Predicted Rating</span>
                            </div>
                            <div class="confidence-interval">
                                <span class="ci-value">[${rec.confidence_interval[0]}, ${rec.confidence_interval[1]}]</span>
                                <span class="ci-label">95% Confidence Interval</span>
                            </div>
                            <div class="uncertainty">
                                <span class="uncertainty-value">±${rec.uncertainty}</span>
                                <span class="uncertainty-label">Margin of Error (95% CI)</span>
                                <div class="uncertainty-bar">
                                    <div class="uncertainty-fill" style="width: ${Math.min(rec.uncertainty * 10, 100)}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    resultsDiv.innerHTML = resultsHTML;
    showResults();
}

function showLoading(show) {
    loadingDiv.classList.toggle('hidden', !show);
}

function showResults() {
    resultsDiv.classList.remove('hidden');
}

function hideResults() {
    resultsDiv.classList.add('hidden');
}

function showNotice(message, type = 'info') {
    noticeDiv.className = `notice ${type}`;
    noticeDiv.textContent = message;
    noticeDiv.classList.remove('hidden');
}

function hideNotice() {
    noticeDiv.classList.add('hidden');
}