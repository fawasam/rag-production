document.addEventListener('DOMContentLoaded', () => {
    // Navigation Routing & Views
    const navItems = document.querySelectorAll('.nav-item[data-target]');
    const viewContents = document.querySelectorAll('.view-content');
    const pageTitle = document.getElementById('pageTitle');
    const pageSubtitle = document.getElementById('pageSubtitle');
    const btnNewQuery = document.getElementById('btnNewQuery');
    const btnViewAllQueries = document.getElementById('btnViewAllQueries');

    function switchView(targetId, titleText, subtitleText) {
        viewContents.forEach(v => v.classList.remove('active'));
        navItems.forEach(n => n.classList.remove('active'));

        const targetView = document.getElementById(targetId);
        if (targetView) targetView.classList.add('active');

        const activeNavItem = document.querySelector(`.nav-item[data-target="${targetId}"]`);
        if (activeNavItem) activeNavItem.classList.add('active');

        pageTitle.textContent = titleText || 'RAG Console';
        pageSubtitle.textContent = subtitleText || 'Production Overview';

        if (targetId === 'logsView') {
            fetchAnalytics();
        } else if (targetId === 'uploadView') {
            fetchExistingDocuments();
        }
    }

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetId = item.getAttribute('data-target');
            let title = 'RAG Console';
            let subtitle = 'Production Overview';

            if (targetId === 'playgroundView') {
                title = 'Query Playground';
                subtitle = 'Test & Inspect Grounded Generation';
            } else if (targetId === 'uploadView') {
                title = 'Document Upload & Ingestion';
                subtitle = 'Asynchronous File Ingestion & Auto-Watcher';
            } else if (targetId === 'logsView') {
                title = 'Query Logs & Analytics';
                subtitle = 'Production Telemetry Records';
            }

            switchView(targetId, title, subtitle);
        });
    });

    btnNewQuery.addEventListener('click', () => {
        switchView('playgroundView', 'Query Playground', 'Test & Inspect Grounded Generation');
        document.getElementById('queryInput').focus();
    });

    if (btnViewAllQueries) {
        btnViewAllQueries.addEventListener('click', () => {
            switchView('logsView', 'Query Logs & Analytics', 'Production Telemetry Records');
        });
    }

    // Chart.js Instances
    let queryTrendsChart = null;
    let dataSourcesChart = null;

    function initChartCanvases() {
        const ctxTrends = document.getElementById('queryTrendsChart');
        if (ctxTrends) {
            queryTrendsChart = new Chart(ctxTrends, {
                type: 'line',
                data: {
                    labels: ['Run 1'],
                    datasets: [
                        {
                            label: 'Latency (s)',
                            data: [0],
                            borderColor: '#6C5CE7',
                            backgroundColor: 'rgba(108, 92, 231, 0.08)',
                            tension: 0.3,
                            fill: true,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Grounded (%)',
                            data: [100],
                            borderColor: '#10B981',
                            borderDash: [4, 4],
                            tension: 0.3,
                            fill: false,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { position: 'top', align: 'start', labels: { boxWidth: 12, font: { family: 'Inter' } } }
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: 'Latency (seconds)' },
                            grid: { color: '#F1F5F9' }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            min: 0,
                            max: 100,
                            grid: { drawOnChartArea: false },
                            ticks: { callback: v => v + '%' }
                        },
                        x: { grid: { display: false } }
                    }
                }
            });
        }

        const ctxDonut = document.getElementById('dataSourcesChart');
        if (ctxDonut) {
            dataSourcesChart = new Chart(ctxDonut, {
                type: 'doughnut',
                data: {
                    labels: ['Loading...'],
                    datasets: [{
                        data: [1],
                        backgroundColor: ['#6C5CE7', '#10B981', '#3B82F6', '#F59E0B', '#94A3B8'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'right', labels: { boxWidth: 10, font: { family: 'Inter', size: 11 } } }
                    },
                    cutout: '70%'
                }
            });
        }
    }
    initChartCanvases();

    // Fetch Real System Telemetry & Documents Data
    async function loadRealTelemetry() {
        try {
            const resp = await fetch('/v1/logs?limit=20', {
                headers: { 'x-api-key': 'X9usnoG4t0zcAujbzwEqhVllp_5LbKHKR3Tzn05U4zo' }
            });
            if (!resp.ok) return;

            const data = await resp.json();
            const summary = data.summary || {};
            const logs = data.logs || [];

            // Update KPI Cards with Real Data
            document.getElementById('kpiTotalQueries').textContent = summary.total_queries || 0;
            document.getElementById('kpiSuccessRate').textContent = `${summary.citation_valid_rate || 100}%`;
            document.getElementById('kpiAvgResponse').textContent = `${summary.avg_latency_seconds || 0}s`;
            document.getElementById('kpiTotalDocs').textContent = summary.total_docs || 0;
            document.getElementById('kpiTotalChunksFooter').textContent = `${summary.total_chunks || 0} total vector chunks`;

            // Update Donut Chart with Real Document Formats
            if (dataSourcesChart && summary.file_formats) {
                const labels = Object.keys(summary.file_formats);
                const values = Object.values(summary.file_formats);
                if (labels.length > 0) {
                    dataSourcesChart.data.labels = labels;
                    dataSourcesChart.data.datasets[0].data = values;
                    dataSourcesChart.update();
                }
            }

            // Update Line Chart with Real Query Timeline
            if (queryTrendsChart && logs.length > 0) {
                const chronologicalLogs = [...logs].reverse();
                const labels = chronologicalLogs.map((l, idx) => {
                    return l.timestamp ? new Date(l.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : `#${idx + 1}`;
                });
                const latencies = chronologicalLogs.map(l => l.latency_seconds || 0);
                const groundedFlags = chronologicalLogs.map(l => l.citations_valid ? 100 : 0);

                queryTrendsChart.data.labels = labels;
                queryTrendsChart.data.datasets[0].data = latencies;
                queryTrendsChart.data.datasets[1].data = groundedFlags;
                queryTrendsChart.update();
            }

            // Update Recent Queries Feed
            renderRecentFeed(logs.slice(0, 5));

        } catch (err) {
            console.error('Error fetching real telemetry:', err);
        }
    }
    loadRealTelemetry();

    function renderRecentFeed(logs) {
        const feedContainer = document.getElementById('recentQueriesList');
        if (!feedContainer) return;

        if (!logs || logs.length === 0) {
            feedContainer.innerHTML = '<div class="subtext">No recent queries logged yet</div>';
            return;
        }

        feedContainer.innerHTML = logs.map(log => {
            const statusClass = log.citations_valid ? 'success' : 'partial';
            const statusText = log.citations_valid ? 'Success' : 'Partial';
            return `
                <div class="query-feed-item">
                    <div class="feed-query-text" title="${escapeHtml(log.query)}">💬 ${escapeHtml(log.query)}</div>
                    <div class="feed-meta">
                        <span class="feed-latency">${log.latency_seconds || 0}s</span>
                        <span class="badge-status ${statusClass}">${statusText}</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    // Playground Controls & Form Logic
    const apiKeyInput = document.getElementById('apiKeyInput');
    const toggleKeyVisibility = document.getElementById('toggleKeyVisibility');
    const topKSlider = document.getElementById('topKSlider');
    const topKValue = document.getElementById('topKValue');

    const queryForm = document.getElementById('queryForm');
    const queryInput = document.getElementById('queryInput');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn?.querySelector('.btn-text');
    const btnSpinner = submitBtn?.querySelector('.btn-spinner');

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

    if (toggleKeyVisibility) {
        toggleKeyVisibility.addEventListener('click', () => {
            const type = apiKeyInput.getAttribute('type') === 'password' ? 'text' : 'password';
            apiKeyInput.setAttribute('type', type);
            toggleKeyVisibility.textContent = type === 'password' ? '👁️' : '🙈';
        });
    }

    if (topKSlider) {
        topKSlider.addEventListener('input', (e) => {
            topKValue.textContent = e.target.value;
        });
    }

    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            queryInput.value = chip.getAttribute('data-query');
            queryForm.dispatchEvent(new Event('submit', { cancelable: true }));
        });
    });

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

    if (queryForm) {
        queryForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const query = queryInput.value.trim();
            if (!query) return;

            submitBtn.disabled = true;
            if (btnText) btnText.textContent = 'Processing...';
            if (btnSpinner) btnSpinner.classList.remove('hidden');

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

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `HTTP Error ${response.status}`);
                }

                const data = await response.json();
                renderResults(data);
                loadRealTelemetry();

            } catch (err) {
                alert(`Query Failed: ${err.message}`);
            } finally {
                submitBtn.disabled = false;
                if (btnText) btnText.textContent = 'Run Query';
                if (btnSpinner) btnSpinner.classList.add('hidden');
            }
        });
    }

    function renderResults(data) {
        resultCard.classList.remove('hidden');
        inspectorCard.classList.remove('hidden');

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

        answerText.textContent = data.answer;

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
                <span class="chunk-index">#${idx + 1}</span>
                <span class="chunk-id">${escapeHtml(chunkId)}</span>
            </div>
        `).join('');
    }

    // Analytics View Loader
    const btnRefreshLogs = document.getElementById('btnRefreshLogs');
    const logsTableBody = document.getElementById('logsTableBody');
    const statTotalQueries = document.getElementById('statTotalQueries');
    const statAvgLatency = document.getElementById('statAvgLatency');
    const statCitationAcc = document.getElementById('statCitationAcc');
    const statAnswerableRate = document.getElementById('statAnswerableRate');

    const logModal = document.getElementById('logModal');
    const logModalBody = document.getElementById('logModalBody');
    const btnCloseModal = document.getElementById('btnCloseModal');

    if (btnRefreshLogs) {
        btnRefreshLogs.addEventListener('click', fetchAnalytics);
    }

    async function fetchAnalytics() {
        if (!logsTableBody) return;
        logsTableBody.innerHTML = '<tr><td colspan="6" class="text-center">Loading execution logs...</td></tr>';

        try {
            const response = await fetch('/v1/logs?limit=50', {
                headers: { 'x-api-key': apiKeyInput?.value.trim() || 'X9usnoG4t0zcAujbzwEqhVllp_5LbKHKR3Tzn05U4zo' }
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            const summary = data.summary || {};
            const logs = data.logs || [];

            if (statTotalQueries) statTotalQueries.textContent = summary.total_queries || 0;
            if (statAvgLatency) statAvgLatency.textContent = `${summary.avg_latency_seconds || 0}s`;
            if (statCitationAcc) statCitationAcc.textContent = `${summary.citation_valid_rate || 100}%`;
            if (statAnswerableRate) statAnswerableRate.textContent = `${summary.answerable_rate || 100}%`;

            if (logs.length === 0) {
                logsTableBody.innerHTML = '<tr><td colspan="6" class="text-center">No logged queries found.</td></tr>';
                return;
            }

            logsTableBody.innerHTML = logs.map((log, index) => {
                const dateStr = log.timestamp ? new Date(log.timestamp * 1000).toLocaleString() : 'N/A';
                const citValid = log.citations_valid ? '<span class="badge-status success">✓ Valid</span>' : '<span class="badge-status partial">⚠️ Warning</span>';
                const answerable = log.answerable ? '<span class="badge-status success">True</span>' : '<span class="badge-status partial">False</span>';

                return `
                    <tr>
                        <td>${escapeHtml(dateStr)}</td>
                        <td style="font-weight:500;" title="${escapeHtml(log.query)}">${escapeHtml(log.query)}</td>
                        <td>${log.latency_seconds || 0}s</td>
                        <td>${answerable}</td>
                        <td>${citValid}</td>
                        <td>
                            <button class="pill-btn btn-inspect-log" data-index="${index}">Inspect</button>
                        </td>
                    </tr>
                `;
            }).join('');

            document.querySelectorAll('.btn-inspect-log').forEach(btn => {
                btn.addEventListener('click', () => {
                    const idx = parseInt(btn.getAttribute('data-index'), 10);
                    showLogModal(logs[idx]);
                });
            });

        } catch (err) {
            logsTableBody.innerHTML = `<tr><td colspan="6" class="text-center" style="color:var(--accent-rose);">Failed to load logs: ${escapeHtml(err.message)}</td></tr>`;
        }
    }

    if (btnCloseModal) {
        btnCloseModal.addEventListener('click', () => logModal.classList.add('hidden'));
    }
    if (logModal) {
        logModal.addEventListener('click', (e) => {
            if (e.target === logModal) logModal.classList.add('hidden');
        });
    }

    function showLogModal(log) {
        if (!log || !logModalBody) return;
        logModalBody.innerHTML = `
            <div class="form-group">
                <label>Query Text</label>
                <div style="font-weight:600; color:var(--text-primary); margin-bottom:12px;">${escapeHtml(log.query)}</div>
            </div>
            <div class="form-group">
                <label>Generated Answer</label>
                <div style="background:#F8FAFC; padding:12px; border-radius:var(--radius-sm); font-size:0.9rem; margin-bottom:12px; white-space:pre-wrap;">${escapeHtml(log.answer)}</div>
            </div>
            <div class="form-group">
                <label>Retrieved Chunks (${log.retrieval_debug?.reranked?.length || 0})</label>
                <div style="font-family:var(--font-mono); font-size:0.8rem; color:var(--primary-purple);">
                    ${(log.retrieval_debug?.reranked || []).map(id => `<div>📄 ${escapeHtml(id)}</div>`).join('')}
                </div>
            </div>
        `;
        logModal.classList.remove('hidden');
    }

    // Upload Modal Handlers
    const btnOpenUploadModal = document.getElementById('btnOpenUploadModal');
    const btnCloseUploadModal = document.getElementById('btnCloseUploadModal');
    const uploadModal = document.getElementById('uploadModal');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const uploadStatus = document.getElementById('uploadStatus');

    if (btnOpenUploadModal && uploadModal) {
        btnOpenUploadModal.addEventListener('click', () => {
            uploadStatus.className = 'upload-status hidden';
            uploadModal.classList.remove('hidden');
        });
    }

    if (btnCloseUploadModal && uploadModal) {
        btnCloseUploadModal.addEventListener('click', () => {
            uploadModal.classList.add('hidden');
        });
    }

    if (dropzone && fileInput) {
        dropzone.addEventListener('click', () => fileInput.click());

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                const file = e.dataTransfer.files[0];
                dropzone.querySelector('.dropzone-text').innerHTML = `Selected file: <strong>${escapeHtml(file.name)}</strong> (${Math.round(file.size / 1024)} KB)`;
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                dropzone.querySelector('.dropzone-text').innerHTML = `Selected file: <strong>${escapeHtml(file.name)}</strong> (${Math.round(file.size / 1024)} KB)`;
            }
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!fileInput.files || fileInput.files.length === 0) return;

            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);

            uploadStatus.className = 'upload-status';
            uploadStatus.textContent = 'Uploading and scheduling background re-indexing...';
            uploadStatus.classList.remove('hidden');

            try {
                const response = await fetch('/v1/ingest', {
                    method: 'POST',
                    headers: {
                        'x-api-key': apiKeyInput?.value.trim() || 'X9usnoG4t0zcAujbzwEqhVllp_5LbKHKR3Tzn05U4zo'
                    },
                    body: formData
                });

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || `Upload failed with status ${response.status}`);
                }

                uploadStatus.className = 'upload-status success';
                uploadStatus.textContent = `✓ ${data.message || 'File uploaded successfully!'}`;

                loadRealTelemetry();
                setTimeout(() => {
                    uploadModal.classList.add('hidden');
                }, 2000);

            } catch (err) {
                uploadStatus.className = 'upload-status error';
                uploadStatus.textContent = `❌ Upload failed: ${err.message}`;
            }
        });
    }

    // View Upload Dropzone Handlers
    const viewDropzone = document.getElementById('viewDropzone');
    const viewFileInput = document.getElementById('viewFileInput');
    const viewUploadForm = document.getElementById('viewUploadForm');
    const viewUploadStatus = document.getElementById('viewUploadStatus');

    if (viewDropzone && viewFileInput) {
        viewDropzone.addEventListener('click', () => viewFileInput.click());

        viewDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            viewDropzone.classList.add('dragover');
        });

        viewDropzone.addEventListener('dragleave', () => {
            viewDropzone.classList.remove('dragover');
        });

        viewDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            viewDropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                viewFileInput.files = e.dataTransfer.files;
                const file = e.dataTransfer.files[0];
                viewDropzone.querySelector('.dropzone-text').innerHTML = `Selected file: <strong>${escapeHtml(file.name)}</strong> (${Math.round(file.size / 1024)} KB)`;
            }
        });

        viewFileInput.addEventListener('change', () => {
            if (viewFileInput.files.length > 0) {
                const file = viewFileInput.files[0];
                viewDropzone.querySelector('.dropzone-text').innerHTML = `Selected file: <strong>${escapeHtml(file.name)}</strong> (${Math.round(file.size / 1024)} KB)`;
            }
        });
    }

    if (viewUploadForm) {
        viewUploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!viewFileInput.files || viewFileInput.files.length === 0) return;

            const file = viewFileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);

            viewUploadStatus.className = 'upload-status';
            viewUploadStatus.textContent = 'Uploading and scheduling background re-indexing...';
            viewUploadStatus.classList.remove('hidden');

            try {
                const response = await fetch('/v1/ingest', {
                    method: 'POST',
                    headers: {
                        'x-api-key': apiKeyInput?.value.trim() || 'X9usnoG4t0zcAujbzwEqhVllp_5LbKHKR3Tzn05U4zo'
                    },
                    body: formData
                });

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || `Upload failed with status ${response.status}`);
                }

                viewUploadStatus.className = 'upload-status success';
                viewUploadStatus.textContent = `✓ ${data.message || 'File uploaded successfully!'}`;
                loadRealTelemetry();
                fetchExistingDocuments();

            } catch (err) {
                viewUploadStatus.className = 'upload-status error';
                viewUploadStatus.textContent = `❌ Upload failed: ${err.message}`;
            }
        });
    }

    // Existing Documents Table Loader
    const btnRefreshDocs = document.getElementById('btnRefreshDocs');
    const documentsTableBody = document.getElementById('documentsTableBody');
    const existingDocCount = document.getElementById('existingDocCount');

    if (btnRefreshDocs) {
        btnRefreshDocs.addEventListener('click', fetchExistingDocuments);
    }

    async function fetchExistingDocuments() {
        if (!documentsTableBody) return;
        documentsTableBody.innerHTML = '<tr><td colspan="5" class="text-center">Loading existing documents...</td></tr>';

        try {
            const response = await fetch('/v1/documents', {
                headers: { 'x-api-key': apiKeyInput?.value.trim() || 'X9usnoG4t0zcAujbzwEqhVllp_5LbKHKR3Tzn05U4zo' }
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            const docs = data.documents || [];

            if (existingDocCount) existingDocCount.textContent = data.total || 0;

            if (docs.length === 0) {
                documentsTableBody.innerHTML = '<tr><td colspan="5" class="text-center">No raw documents stored in data/raw/ yet.</td></tr>';
                return;
            }

            documentsTableBody.innerHTML = docs.map(doc => {
                const formatClass = doc.format.toLowerCase() === 'pdf' ? 'purple' : 'success';
                const sizeKb = Math.round(doc.size_bytes / 1024);
                const sizeStr = sizeKb > 1024 ? `${(sizeKb / 1024).toFixed(1)} MB` : `${sizeKb} KB`;

                return `
                    <tr>
                        <td style="font-weight:500;" title="${escapeHtml(doc.filename)}">📄 ${escapeHtml(doc.filename)}</td>
                        <td><span class="badge-status ${formatClass}">${escapeHtml(doc.format)}</span></td>
                        <td>${sizeStr}</td>
                        <td><span style="font-weight:600; color:var(--primary-purple);">${doc.chunk_count}</span> chunks</td>
                        <td>
                            <button class="pill-btn btn-delete-doc" data-filename="${escapeHtml(doc.filename)}" style="color:var(--accent-rose);">Delete</button>
                        </td>
                    </tr>
                `;
            }).join('');

            document.querySelectorAll('.btn-delete-doc').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const filename = btn.getAttribute('data-filename');
                    if (confirm(`Are you sure you want to delete '${filename}'?`)) {
                        await deleteDocument(filename);
                    }
                });
            });

        } catch (err) {
            documentsTableBody.innerHTML = `<tr><td colspan="5" class="text-center" style="color:var(--accent-rose);">Failed to load documents: ${escapeHtml(err.message)}</td></tr>`;
        }
    }

    async function deleteDocument(filename) {
        try {
            const response = await fetch(`/v1/documents/${encodeURIComponent(filename)}`, {
                method: 'DELETE',
                headers: { 'x-api-key': apiKeyInput?.value.trim() || 'X9usnoG4t0zcAujbzwEqhVllp_5LbKHKR3Tzn05U4zo' }
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${response.status}`);
            }

            fetchExistingDocuments();
            loadRealTelemetry();
        } catch (err) {
            alert(`Failed to delete document: ${err.message}`);
        }
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }
});



