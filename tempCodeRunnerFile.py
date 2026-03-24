const express = require('express');
const cors = require('cors');
const path = require('path');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Serve static files (index.html, script.js, style.css) from project root
app.use(express.static(path.join(__dirname)));

// Health check
app.get('/health', (req, res) => {
	res.json({ status: 'ok' });
});

// Helper: delay
function delayMs(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

// Simple in-memory cache for prompt -> response (short TTL to reduce re-hits)
const CACHE_TTL_MS = 60_000;
const MAX_CACHE_SIZE = 1000; // Prevent unlimited memory growth
const promptCache = new Map(); // key: prompt, value: { text, expiresAt }

function getCachedResponse(prompt) {
	const entry = promptCache.get(prompt);
	if (entry && Date.now() < entry.expiresAt) {
		return entry.text;
	}
	if (entry) promptCache.delete(prompt);
	return null;
}

function setCachedResponse(prompt, text) {
	// Clean expired entries periodically
	if (promptCache.size > MAX_CACHE_SIZE) {
		const now = Date.now();
		for (const [key, value] of promptCache.entries()) {
			if (now >= value.expiresAt) {
				promptCache.delete(key);
			}
		}
	}
	
	promptCache.set(prompt, { text, expiresAt: Date.now() + CACHE_TTL_MS });
}

// Rate limiting to prevent API overload while allowing some concurrency
const MAX_CONCURRENT_REQUESTS = 3;
let activeRequests = 0;
const requestQueue = [];

function enqueueRequest(task) {
	return new Promise((resolve, reject) => {
		requestQueue.push({ task, resolve, reject });
		processQueue();
	});
}

function processQueue() {
	if (activeRequests >= MAX_CONCURRENT_REQUESTS || requestQueue.length === 0) {
		return;
	}
	
	const { task, resolve, reject } = requestQueue.shift();
	activeRequests++;
	
	task()
		.then(resolve)
		.catch(reject)
		.finally(() => {
			activeRequests--;
			processQueue();
		});
}

// Real AI endpoint using Google Gemini with timeout and retries
app.post('/api/gemini', async (req, res) => {
	try {
		const { prompt } = req.body || {};
		if (!prompt || typeof prompt !== 'string') {
			return res.status(400).json({ error: 'Invalid prompt' });
		}

		// Serve from cache if we have a very recent identical prompt
		const cached = getCachedResponse(prompt);
		if (cached) {
			return res.json({ response: cached, cached: true });
		}

		const apiKey = process.env.GEMINI_API_KEY;
		if (!apiKey) {
			return res.status(500).json({ error: 'Server misconfiguration: GEMINI_API_KEY is missing' });
		}

		const url = 'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent';

		// Request config: slightly higher timeout to reduce client timeouts
		const axiosConfig = {
			timeout: 12000,
			headers: { 'Content-Type': 'application/json' },
		};

		// Enhanced prompt with price range instructions
		const enhancedPrompt = `You are Yatri, a helpful travel and product assistant. When users ask about product prices or price ranges, always provide approximate price ranges based on current market data. Never say you don't know or can't provide pricing information. Instead, give reasonable price estimates for:

- Electronics: Provide typical price ranges (e.g., smartphones: $200-$1500, laptops: $300-$3000)
- Clothing: Give approximate ranges (e.g., t-shirts: $10-$50, jeans: $30-$150)
- Food items: Provide local market estimates (e.g., bread: $2-$5, milk: $3-$6)
- Travel items: Give typical costs (e.g., backpacks: $30-$200, suitcases: $50-$400)
- Any other products: Always provide reasonable price estimates

User question: ${prompt}`;

		const payload = {
			contents: [
				{ parts: [{ text: enhancedPrompt }] },
			],
		};

		let lastError = null;
		const maxAttempts = 2; // quick retry once

		// Rate limit upstream calls to prevent overload
		const result = await enqueueRequest(async () => {
			for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
				try {
					const { data } = await axios.post(${url}?key=${apiKey}, payload, axiosConfig);
					const text = data?.candidates?.[0]?.content?.parts?.[0]?.text || 'Sorry, I could not generate a response.';
					setCachedResponse(prompt, text);
					return { ok: true, text };
				} catch (err) {
					lastError = err;
					const status = err?.response?.status;
					// Retry on transient errors
					if (attempt < maxAttempts && (status === 429 || status === 500 || status === 502 || status === 503 || status === 504)) {
						await delayMs(400 * attempt);
						continue;
					}
					break;
				}
			}
			return { ok: false };
		});

		if (result.ok) {
			return res.json({ response: result.text });
		}

		console.error('Error in /api/gemini:', lastError?.response?.data || lastError?.message || lastError);
		console.error('Full error details:', JSON.stringify(lastError?.response?.data, null, 2));
		return res.status(200).json({ response: 'I am experiencing high load right now. Please try again in a moment.' });
	} catch (error) {
		console.error('Error in /api/gemini (outer):', error);
		const status = error?.response?.status || 500;
		res.status(status).json({ error: 'Failed to get AI response' });
	}
});

// Fallback to index.html for root
app.get('/', (req, res) => {
	res.sendFile(path.join(__dirname, 'index.html'));
});

const server = app.listen(PORT, () => {
	console.log(Server running on http://localhost:${PORT});
});

// Graceful shutdown to release the port cleanly on restarts (nodemon)
const shutdown = (signal) => {
	try {
		console.log(Received ${signal}. Shutting down gracefully...);
		server.close(() => {
			console.log('HTTP server closed');
			process.exit(0);
		});
		// Force exit if not closed in time
		setTimeout(() => {
			console.warn('Forcing shutdown');
			process.exit(1);
		}, 3000).unref();
	} catch {
		process.exit(1);
	}
};

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));