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

// Tab elements
const datasetUsersTab = document.getElementById('datasetUsersTab');
const newUsersTab = document.getElementById('newUsersTab');
const datasetUsersContent = document.getElementById('datasetUsersContent');
const newUsersContent = document.getElementById('newUsersContent');

// Fuzzy search elements
const bookSearchInput = document.getElementById('bookSearch');
const searchSuggestionsDiv = document.getElementById('searchSuggestions');
const userRatingsDiv = document.getElementById('userRatings');
const fuzzyTopKSlider = document.getElementById('fuzzyTopK');
const fuzzyTopKValue = document.getElementById('fuzzyTopKValue');
const getFuzzyRecsBtn = document.getElementById('getFuzzyRecs');
const fuzzyControls = document.getElementById('fuzzyControls');

// State
let availableUsers = [];
let userRatings = [];
let searchTimeout;
let allBooks = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
});

function setupEventListeners() {
    // Debug: check all elements
    console.log('Element check:', {
        datasetUsersTab,
        newUsersTab,
        topKSlider,
        fuzzyTopKSlider,
        bookSearchInput
    });

    // Tab switching
    if (datasetUsersTab && newUsersTab) {
        datasetUsersTab.addEventListener('click', () => {
            console.log('Dataset Users tab clicked');
            switchTab('dataset');
        });
        newUsersTab.addEventListener('click', () => {
            console.log('New Users tab clicked');  
            switchTab('newUsers');
        });
    } else {
        console.error('Tab elements not found:', { datasetUsersTab, newUsersTab });
    }
    
    // Update slider value display - with null checks
    if (topKSlider && topKValue) {
        topKSlider.addEventListener('input', () => {
            topKValue.textContent = topKSlider.value;
        });
    }
    
    if (fuzzyTopKSlider && fuzzyTopKValue) {
        fuzzyTopKSlider.addEventListener('input', () => {
            fuzzyTopKValue.textContent = fuzzyTopKSlider.value;
        });
    }

    // Get recommendations - with null checks
    if (getRecommendationsBtn) {
        getRecommendationsBtn.addEventListener('click', getRecommendations);
    }
    if (getFuzzyRecsBtn) {
        getFuzzyRecsBtn.addEventListener('click', getColdStartRecommendations);
    }
    
    // Random user
    if (randomUserBtn) {
        randomUserBtn.addEventListener('click', selectRandomUser);
    }
    
    // Real-time book search with auto-complete
    if (bookSearchInput) {
        bookSearchInput.addEventListener('input', handleSearchInput);
        bookSearchInput.addEventListener('blur', () => {
            // Hide suggestions after a short delay to allow clicking
            setTimeout(() => hideSuggestions(), 200);
        });
        bookSearchInput.addEventListener('focus', () => {
            if (bookSearchInput.value.length >= 3) {
                showSuggestions(bookSearchInput.value);
            }
        });
    }
    
    // Enter key on user input
    if (userIdInput) {
        userIdInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                getRecommendations();
            }
        });
    }
    
    // Hide suggestions when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-container')) {
            hideSuggestions();
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
        
        // Load all books for search
        await loadAllBooks();
        
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
                    <span class="stat-label">Total Users with Recommendations:</span>
                    <span class="stat-value">${stats.total_users_with_recs}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Memory Usage:</span>
                    <span class="stat-value">${stats.memory_usage_mb} MB</span>
                </div>
            </div>
            <p style="font-size: 0.8rem; color: #7f8c8d; margin-top: 1rem;">
                Includes both trained users and cold start users from original dataset
            </p>
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
        // Automatically trigger recommendations
        getRecommendations();
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


// Tab switching
function switchTab(tab) {
    console.log('Switching to tab:', tab);
    
    try {
        if (tab === 'dataset') {
            // Activate Dataset Users tab
            datasetUsersTab.classList.add('active');
            newUsersTab.classList.remove('active');
            datasetUsersContent.classList.remove('hidden');
            newUsersContent.classList.add('hidden');
            console.log('Switched to dataset tab');
        } else if (tab === 'newUsers') {
            // Activate New Users tab  
            newUsersTab.classList.add('active');
            datasetUsersTab.classList.remove('active');
            newUsersContent.classList.remove('hidden');
            datasetUsersContent.classList.add('hidden');
            console.log('Switched to new users tab');
        }
        
        // Clear results when switching tabs
        hideResults();
        hideNotice();
    } catch (error) {
        console.error('Error switching tabs:', error);
    }
}

// Load all books for auto-complete
async function loadAllBooks() {
    try {
        const response = await fetch(`${API_BASE_URL}/books?limit=6000`);
        const data = await response.json();
        allBooks = data.books || [];
        console.log(`Loaded ${allBooks.length} books for search`);
    } catch (error) {
        console.error('Failed to load books:', error);
        allBooks = [];
    }
}

// Auto-complete search functionality
function handleSearchInput(e) {
    const query = e.target.value.trim();
    
    // Clear previous timeout
    if (searchTimeout) {
        clearTimeout(searchTimeout);
    }
    
    if (query.length < 3) {
        hideSuggestions();
        return;
    }
    
    // Debounce search for better performance
    searchTimeout = setTimeout(() => {
        showSuggestions(query);
    }, 150);
}

function showSuggestions(query) {
    if (allBooks.length === 0) {
        return;
    }
    
    // Simple fuzzy search - find books containing the query words
    const queryWords = query.toLowerCase().split(' ').filter(w => w.length > 0);
    
    const matches = allBooks.filter(book => {
        const bookTitle = book.toLowerCase();
        return queryWords.every(word => bookTitle.includes(word));
    }).slice(0, 8); // Show top 8 matches
    
    if (matches.length === 0) {
        hideSuggestions();
        return;
    }
    
    const suggestionsHTML = matches.map(book => `
        <div class="suggestion-item" onclick="addBookToRatings('${book.replace(/'/g, "\\'")}')">
            <span class="suggestion-title">${book}</span>
            <button class="suggestion-add-btn" onclick="event.stopPropagation(); addBookToRatings('${book.replace(/'/g, "\\'")}')">
                Add & Rate
            </button>
        </div>
    `).join('');
    
    searchSuggestionsDiv.innerHTML = suggestionsHTML;
    searchSuggestionsDiv.classList.remove('hidden');
}

function hideSuggestions() {
    searchSuggestionsDiv.classList.add('hidden');
}

function addBookToRatings(bookTitle) {
    // Check if book is already in ratings
    const existing = userRatings.find(r => r.title === bookTitle);
    if (existing) {
        showNotice('Book already in your ratings list', 'warning');
        return;
    }
    
    // Add book with default rating
    userRatings.push({
        title: bookTitle,
        rating: 7 // Default rating
    });
    
    updateRatingsDisplay();
    updateRecommendationsButton();
    
    // Clear search and hide suggestions
    bookSearchInput.value = '';
    hideSuggestions();
    
    showNotice(`Added "${bookTitle}" to your ratings`, 'info');
}

function updateRatingsDisplay() {
    if (userRatings.length === 0) {
        userRatingsDiv.innerHTML = '';
        return;
    }
    
    const ratingsHTML = `
        <h4>Your Book Ratings (${userRatings.length})</h4>
        ${userRatings.map((rating, index) => `
            <div class="rating-item">
                <div class="book-title-rating">${rating.title}</div>
                <div class="rating-controls">
                    <input type="number" class="rating-input" min="1" max="10" 
                           value="${rating.rating}" 
                           onchange="updateRating(${index}, this.value)">
                    <span>/10</span>
                    <button class="remove-rating" onclick="removeRating(${index})">Remove</button>
                </div>
            </div>
        `).join('')}
    `;
    
    userRatingsDiv.innerHTML = ratingsHTML;
}

function updateRating(index, newRating) {
    const rating = parseFloat(newRating);
    if (rating >= 1 && rating <= 10) {
        userRatings[index].rating = rating;
    }
}

function removeRating(index) {
    const removedBook = userRatings[index].title;
    userRatings.splice(index, 1);
    updateRatingsDisplay();
    updateRecommendationsButton();
    showNotice(`Removed "${removedBook}" from your ratings`, 'info');
}

function updateRecommendationsButton() {
    getFuzzyRecsBtn.disabled = userRatings.length === 0;
    if (userRatings.length > 0) {
        fuzzyControls.style.display = 'flex';
    } else {
        fuzzyControls.style.display = 'none';
    }
}

// Cold start recommendations
async function getColdStartRecommendations() {
    if (userRatings.length === 0) {
        showNotice('Please add and rate at least one book first', 'warning');
        return;
    }
    
    const topK = fuzzyTopKSlider.value;
    
    showLoading(true);
    hideNotice();
    hideResults();
    
    try {
        const response = await fetch(`${API_BASE_URL}/recommend/cold-start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                ratings: userRatings,
                top_k: parseInt(topK)
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get recommendations');
        }
        
        const data = await response.json();
        displayColdStartRecommendations(data);
        
        if (data.note) {
            showNotice(data.note, 'info');
        }
        
    } catch (error) {
        showNotice(`Error: ${error.message}`, 'error');
        console.error('Cold start recommendation error:', error);
    } finally {
        showLoading(false);
    }
}

function displayColdStartRecommendations(data) {
    const recommendations = data.recommendations;
    
    if (recommendations.length === 0) {
        showNotice('No recommendations could be generated from your ratings', 'info');
        return;
    }
    
    const resultsHTML = `
        <div class="results-header">
            <h2>Your Personalized Recommendations</h2>
            <p>Based on ${data.valid_books_rated} books you rated from our dataset</p>
            <p>Showing ${recommendations.length} recommendations</p>
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
                                    <div class="uncertainty-fill" style="width: ${Math.min(rec.uncertainty * 20, 100)}%"></div>
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