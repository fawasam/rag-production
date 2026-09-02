document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const latencyBadge = document.getElementById('latencyBadge');
    const latencyVal = document.getElementById('latencyVal');

    const apiKeyInput = document.getElementById('apiKeyInput');
    const toggleKeyVisibility = document.getElementById('toggleKeyVisibility');
    const topKSlider = document.getElementById('topKSlider');
    const topKValue = document.getElementById('topKValue');

    const queryForm = document.getElementById('queryForm');
    const queryInput = document.getElementById('queryInput');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnSpinner = submitBtn.querySelector('.btn-spinner');

    const resultCard = document.getElementById('resultCard');
    const groundedBadge = document.getElementById('groundedBadge');
    const answerableBadge = document.getElementById('answerableBadge');
    const answerText = document.getElementById('answerText');
    const citationsSection = document.getElementById('citationsSection');
    const citationsGrid = document.getElementById('citationsGrid');
    const invalidCitationsBox = document.getElementById('invalidCitationsBox');
    const invalidCitationsList = document.getElementById('invalidCitationsList');

    const inspectorCard = document.getElementById('inspectorCard');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    const denseList = document.getElementById('denseList');
    const bm25List = document.getElementById('bm25List');
    const fusedList = document.getElementById('fusedList');
    const rerankedList = document.getElementById('rerankedList');

    // Health Check Status Polling
    async function checkHealth() {
        try {
            const resp = await fetch('/v1/health');
            if (resp.ok) {
                statusIndicator.className = 'status-indicator online';
                statusText.textContent = 'API Online';
            } else {
                throw new Error('Health check failed');
            }
        } catch (err) {
            statusIndicator.className = 'status-indicator offline';
            statusText.textContent = 'API Offline';
        }
    }
    checkHealth();
    setInterval(checkHealth, 30000);

    // Toggle API Key Visibility
    toggleKeyVisibility.addEventListener('click', () => {
        const type = apiKeyInput.getAttribute('type') === 'password' ? 'text' : 'password';
        apiKeyInput.setAttribute('type', type);
        toggleKeyVisibility.textContent = type === 'password' ? '👁️' : '🙈';
    });

    // Slider Sync
    topKSlider.addEventListener('input', (e) => {
        topKValue.textContent = e.target.value;
    });

    // Sample Chips Handler
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            queryInput.value = chip.getAttribute('data-query');
            queryForm.dispatchEvent(new Event('submit', { cancelable: true }));
        });
    });

    // Inspector Tabs Handler
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetPane = document.getElementById(`tabPane${targetTab.charAt(0).toUpperCase() + targetTab.slice(1)}`);
            if (targetPane) targetPane.classList.add('active');
        });
    });

    // Query Submission Handler
    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // UI Loading State
        submitBtn.disabled = true;
        btnText.textContent = 'Processing...';
        btnSpinner.classList.remove('hidden');

        const startTime = performance.now();

        try {
            const response = await fetch('/v1/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'accept': 'application/json',
                    'x-api-key': apiKeyInput.value.trim()
                },
                body: JSON.stringify({
                    query: query,
                    top_k: parseInt(topKSlider.value, 10)
                })
            });

            const duration = Math.round(performance.now() - startTime);
            latencyVal.textContent = `${duration} ms`;

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP Error ${response.status}`);
            }

            const data = await response.json();
            renderResults(data);

        } catch (err) {
            alert(`Query Failed: ${err.message}`);
        } finally {
            submitBtn.disabled = false;
            btnText.textContent = 'Run Query';
            btnSpinner.classList.add('hidden');
        }
    });

    // Render Query Response
    function renderResults(data) {
        resultCard.classList.remove('hidden');
        inspectorCard.classList.remove('hidden');

        // Render Badges
        if (data.citations_valid) {
            groundedBadge.className = 'grounded-badge valid';
            groundedBadge.textContent = '✓ Grounded';
        } else {
            groundedBadge.className = 'grounded-badge warning';
            groundedBadge.textContent = '⚠️ Citation Warning';
        }

        if (data.answerable) {
            answerableBadge.className = 'answerable-badge true';
            answerableBadge.textContent = 'Answerable';
        } else {
            answerableBadge.className = 'answerable-badge false';
            answerableBadge.textContent = 'Unanswerable';
        }

        // Render Answer Text
        answerText.textContent = data.answer;

        // Render Valid Citations
        if (data.citations && data.citations.length > 0) {
            citationsSection.classList.remove('hidden');
            citationsGrid.innerHTML = data.citations.map(c => `
                <div class="citation-card">
                    <div class="citation-chunk">📄 ${escapeHtml(c.chunk_id)}</div>
                    <div class="citation-quote">"${escapeHtml(c.supporting_quote)}"</div>
                </div>
            `).join('');
        } else {
            citationsSection.classList.add('hidden');
            citationsGrid.innerHTML = '';
        }

        // Render Invalid Citations (if any)
        const invalidCitations = data.debug?.invalid_citations || [];
        if (invalidCitations.length > 0) {
            invalidCitationsBox.classList.remove('hidden');
            invalidCitationsList.innerHTML = invalidCitations.map(ic => `
                <li><strong>${escapeHtml(ic.chunk_id)}:</strong> ${escapeHtml(ic.reason)}</li>
            `).join('');
        } else {
            invalidCitationsBox.classList.add('hidden');
            invalidCitationsList.innerHTML = '';
        }

        // Render Inspector Debug Pipeline Lists
        renderChunkList(denseList, data.debug?.dense || []);
        renderChunkList(bm25List, data.debug?.bm25 || []);
        renderChunkList(fusedList, data.debug?.fused || []);
        renderChunkList(rerankedList, data.retrieved_chunk_ids || data.debug?.reranked || []);
    }

    function renderChunkList(container, chunks) {
        if (!chunks || chunks.length === 0) {
            container.innerHTML = '<div class="subtext">No candidates returned for this stage</div>';
            return;
        }

        container.innerHTML = chunks.map((chunkId, idx) => `
            <div class="chunk-item">
                <div>
                    <span class="chunk-index">#${idx + 1}</span>
                    <span class="chunk-id">${escapeHtml(chunkId)}</span>
                </div>
            </div>
        `).join('');
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }
});
