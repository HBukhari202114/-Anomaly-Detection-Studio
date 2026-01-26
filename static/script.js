document.addEventListener('DOMContentLoaded', () => {
    const fileUpload = document.getElementById('fileUpload');
    const targetColumn = document.getElementById('targetColumn');
    const contaminationInput = document.getElementById('contamination');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const exportBtn = document.getElementById('exportBtn');
    const resultsDiv = document.getElementById('results');
    const detailedResults = document.getElementById('detailedResults');
    const comparisonChartDiv = document.getElementById('comparisonChart');
    const errorDisplay = document.getElementById('error');
    const uploadPreview = document.getElementById('uploadPreview');
    const cancelBtn = document.getElementById('cancelBtn');
    const contaminationSection = document.getElementById('contaminationSection');

    let abortController = null;
    let comparisonChart = null;
    let analysisData = [];

    function toggleContamination() {
        const models = Array.from(document.querySelectorAll('.model-checkbox input:checked'))
                            .map(c => c.value);
        const isUnsupervised = models.some(m => ['isolation_forest', 'autoencoder'].includes(m));
        contaminationSection.classList.toggle('hidden', !isUnsupervised);

        if (!isUnsupervised) {
            contaminationInput.value = '';
        }
    }

    function updateFilePreview() {
        const files = fileUpload.files;
        if (files.length > 0) {
            uploadPreview.textContent = `${files.length} file(s) selected`;
            uploadPreview.classList.remove('hidden');
        } else {
            uploadPreview.classList.add('hidden');
        }
    }

    fileUpload.addEventListener('change', updateFilePreview);
    document.querySelectorAll('.model-checkbox input').forEach(cb => {
        cb.addEventListener('change', toggleContamination);
    });

    analyzeBtn.addEventListener('click', startAnalysis);
    exportBtn.addEventListener('click', exportResults);
    cancelBtn.addEventListener('click', () => {
        if (abortController) {
            abortController.abort();
            analyzeBtn.disabled = false;
            cancelBtn.classList.add('hidden');
            analyzeBtn.textContent = 'Analyze';
        }
    });

    function validateModelSelection() {
        const models = Array.from(document.querySelectorAll('.model-checkbox input:checked'))
                          .map(c => c.value);
        const hasSupervised = models.some(m => ['random_forest', 'lightgbm', 'deep_learning'].includes(m));
        const hasUnsupervised = models.some(m => ['isolation_forest', 'autoencoder'].includes(m));

        contaminationSection.classList.toggle('hidden', !hasUnsupervised);

        if (hasSupervised && !targetColumn.value.trim()) {
            showError('Supervised models require a target column');
            return false;
        }
        return true;
    }

	function showDownloadButton(preprocessedCsvUrl) {
		const downloadSection = document.getElementById("downloadSection");
		const downloadBtn = document.getElementById("downloadBtn");

		if (!preprocessedCsvUrl) {
			console.error("No preprocessed CSV URL provided.");
			return;
		}

		if (downloadSection) {
			downloadSection.classList.remove("hidden");
		}

		if (downloadBtn) {
			downloadBtn.onclick = () => {
				window.location.href = preprocessedCsvUrl;
			};
		}
	}

    async function startAnalysis() {
        if (!validateModelSelection()) return;
        abortController = new AbortController();

        const files = Array.from(fileUpload.files);
        const models = Array.from(document.querySelectorAll('.model-checkbox input:checked'))
                            .map(c => c.value);
        const target = targetColumn.value.trim();
        const contaminationValue = contaminationInput.value.trim();
        const thresholdMode = document.querySelector('input[name="thresholdMode"]:checked')?.value || 'contamination';

        clearResults();
        analyzeBtn.disabled = true;
        cancelBtn.classList.remove('hidden');
        analyzeBtn.textContent = 'Analyzing...';
        errorDisplay.classList.add('hidden');

        try {
            if (!files.length) throw new Error('Please select at least one file');
            if (!models.length) throw new Error('Please select at least one model');

            const formData = new FormData();
            files.forEach(file => formData.append('files', file));
            formData.append('models', models.join(','));
            formData.append('targetColumn', target);
            formData.append('contamination', contaminationValue);
            formData.append('thresholdMode', thresholdMode);

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData,
                signal: abortController.signal
            });

            const data = await response.json();

            if (!response.ok || data.status === 'error') {
                let errorMsg = data.error || 'Analysis failed';
                if (data.errors) {
                    errorMsg += `: ${Array.isArray(data.errors) ? data.errors.join(', ') : data.errors}`;
                }
                throw new Error(errorMsg);
            }

            analysisData = data.results || data;
            renderResults();
            resultsDiv.classList.remove('hidden');

			if (Array.isArray(data) && data.length > 0 && data[0].preprocessed_download) {
				showDownloadButton(data[0].preprocessed_download);
			}

        } catch (error) {
            if (error.name === 'AbortError') {
                showError('Analysis cancelled by user');
            } else {
                showError(error.message);
                console.error('Analysis error:', error);
            }
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze';
            cancelBtn.classList.add('hidden');
            abortController = null;
        }
    }

    function exportResults() {
        if (!analysisData.length) {
            showError('No results to export');
            return;
        }

        let csvContent = "File Name,Model,Accuracy,Precision,Recall,F1-Score,Contamination,% Normal,% Anomalies\n";

        analysisData.forEach(fileResult => {
            Object.entries(fileResult.results).forEach(([model, data]) => {
                csvContent += [
                    `"${fileResult.file_name}"`,
                    `"${model.replace(/_/g, ' ').toUpperCase()}"`,
                    data.metrics.accuracy ? (data.metrics.accuracy * 100).toFixed(2) : 'N/A',
                    data.metrics.precision ? (data.metrics.precision * 100).toFixed(2) : 'N/A',
                    data.metrics.recall ? (data.metrics.recall * 100).toFixed(2) : 'N/A',
                    data.metrics.f1 ? (data.metrics.f1 * 100).toFixed(2) : 'N/A',
                    data.stats.used_contamination || 'Auto',
                    ((data.stats.normal / data.stats.total_samples) * 100).toFixed(2),
                    ((data.stats.anomalies / data.stats.total_samples) * 100).toFixed(2)
                ].join(',') + '\n';
            });
        });

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);

        link.setAttribute('href', url);
        link.setAttribute('download', `anomaly_results_${new Date().toISOString().slice(0,10)}.csv`);
        link.style.visibility = 'hidden';

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

	function getBestModelPerFile(fileResult) {
		let bestModel = null;
		let bestScore = -Infinity;

		Object.entries(fileResult.results).forEach(([modelName, modelData]) => {
			const score = modelData.metrics.f1 || modelData.stats.anomaly_rate;
			if (score > bestScore) {
				bestScore = score;
				bestModel = modelName;
			}
		});
		return bestModel ? bestModel.replace(/_/g, ' ').toUpperCase() : 'N/A';
	}

	function getTotalSamples(fileResult) {
		return Object.values(fileResult.results)[0].stats.total_samples;
	}

	function renderResults() {
		detailedResults.innerHTML = '';
		if (comparisonChart) comparisonChart.destroy();

		const selectedModels = Array.from(document.querySelectorAll('.model-checkbox input:checked'))
								   .map(c => c.value);

		const canvas = document.createElement('canvas');
		comparisonChartDiv.innerHTML = '';
		comparisonChartDiv.appendChild(canvas);
		renderComparisonChart(canvas);

		analysisData.forEach(fileResult => {
			const bestModel = getBestModelPerFile(fileResult);
			const totalSamples = getTotalSamples(fileResult);

			const table = document.createElement('table');
			table.className = 'results-table';

			table.innerHTML = `
				<thead>
					<tr>
						<th colspan="8" class="file-header">
							<span class="file-name">${fileResult.file_name}</span>
							<div class="file-stats">
								<span class="stat-badge">Total Samples: ${totalSamples}</span>
								${selectedModels.length > 1 ? `<span class="stat-badge best-model-badge">Best Model: ${bestModel}</span>` : ''}
							</div>
						</th>
					</tr>
					<tr class="column-headers">
						<th>Model</th>
						<th>Accuracy</th>
						<th>Precision</th>
						<th>Recall</th>
						<th>F1-Score</th>
						<th>Contamination</th>
						<th>% Normal</th>
						<th>% Anomalies</th>
					</tr>
				</thead>
				<tbody>
					${Object.entries(fileResult.results).map(([model, data]) => `
						<tr class="${model === bestModel.toLowerCase().replace(/ /g, '_') ? 'best-model' : ''}" title="Used Contamination: ${(data.stats.used_contamination || 0.1).toFixed(2)}">
							<td>${model.replace(/_/g, ' ').toUpperCase()}</td>
							<td>${data.metrics.accuracy ? `${(data.metrics.accuracy * 100).toFixed(2)}%` : 'N/A'}</td>
							<td>${data.metrics.precision ? `${(data.metrics.precision * 100).toFixed(2)}%` : 'N/A'}</td>
							<td>${data.metrics.recall ? `${(data.metrics.recall * 100).toFixed(2)}%` : 'N/A'}</td>
							<td>${data.metrics.f1 ? `${(data.metrics.f1 * 100).toFixed(2)}%` : 'N/A'}</td>
							<td>${data.stats.used_contamination || 'Auto'}</td>
							<td>${((data.stats.normal / data.stats.total_samples) * 100).toFixed(2)}%</td>
							<td>${((data.stats.anomalies / data.stats.total_samples) * 100).toFixed(2)}%</td>
						</tr>
					`).join('')}
				</tbody>
			`;

			detailedResults.appendChild(table);
		});
	}

	function renderComparisonChart(canvas) {
		const selectedModels = Array.from(document.querySelectorAll('.model-checkbox input:checked'))
								   .map(c => c.value);
		const showBestModel = selectedModels.length > 1;

		const datasets = analysisData.flatMap(fileResult => {
			const bestModel = showBestModel ? getBestModelPerFile(fileResult) : null;
			return Object.entries(fileResult.results).map(([model, data]) => ({
				label: `${model.replace(/_/g, ' ').toUpperCase()}\n(${fileResult.file_name})`,
				total: data.stats.total_samples,
				isBest: model === bestModel?.toLowerCase().replace(/ /g, '_'),
				normal: data.stats.normal,
				anomalies: data.stats.anomalies,
				contamination: data.stats.used_contamination
			}));
		});

		const chartData = {
			labels: datasets.map(d => d.label),
			datasets: [
				{
					label: 'Total Samples',
					type: 'line',
					data: datasets.map(d => d.total),
					borderColor: '#64748b',
					borderWidth: 1,
					borderDash: [5, 5],
					pointRadius: 0,
					order: 1
				},
				...(showBestModel ? [{
					label: 'Best Model',
					type: 'line',
					data: datasets.map(d => d.isBest ? d.total : null),
					borderColor: '#fbbf24',
					borderWidth: 3,
					pointRadius: 0,
					order: 0
				}] : []),
				{
					label: 'Normal',
					data: datasets.map(d => d.normal),
					backgroundColor: '#22c55e',
					borderWidth: 0,
					order: 2
				},
				{
					label: 'Anomalies',
					data: datasets.map(d => d.anomalies),
					backgroundColor: '#ef4444',
					borderWidth: 0,
					order: 2
				}
			]
		};

		comparisonChart = new Chart(canvas.getContext('2d'), {
			type: 'bar',
			data: chartData,
			options: {
				responsive: true,
				maintainAspectRatio: false,
				scales: {
					y: {
						beginAtZero: true,
						ticks: { color: '#9ca3af' },
						title: {
							display: true,
							text: 'Number of Samples',
							color: '#9ca3af'
						}
					},
					x: {
						ticks: {
							color: '#9ca3af',
							maxRotation: 0,
							minRotation: 0
						},
						grid: { display: false }
					}
				},
				plugins: {
					legend: {
						display: true,
						position: 'bottom',
						labels: { color: '#9ca3af' }
					},
					tooltip: {
						callbacks: {
							afterBody: function (tooltipItems) {
								const index = tooltipItems[0].dataIndex;
								const contamination = datasets[index].contamination;
								return `Used Contamination: ${(contamination || 0.1).toFixed(2)}`;
							}
						}
					},
					title: {
						display: true,
						text: 'Normal vs Anomalous Samples',
						color: '#ffffff'
					}
				}
			}
		});
	}

    function clearResults() {
        analysisData = [];
        detailedResults.innerHTML = '';
        comparisonChartDiv.innerHTML = '';
        if (comparisonChart) comparisonChart.destroy();
        document.getElementById("downloadSection").classList.add("hidden");
    }

    function showError(message) {
        errorDisplay.textContent = message;
        errorDisplay.classList.remove('hidden');
        setTimeout(() => errorDisplay.classList.add('hidden'), 5000);
    }

    document.getElementById('autoContam').addEventListener('click', () => {
        contaminationInput.value = '';
    });

    // Typewriter effect for codenesthub footer text
    function initCodenesthubTypewriter() {
        const footerText = document.getElementById('codenesthub-text');
        if (!footerText) return;

        const targetText = 'codenesthub';
        const linkUrl = 'https://codenesthub.vercel.app/';
        let hasAnimated = false;

        // Intersection Observer to detect when footer is in view
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !hasAnimated) {
                    hasAnimated = true;
                    typewriterEffect(footerText, targetText, linkUrl);
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.3 // Trigger when 30% of footer is visible
        });

        // Observe the footer element
        const footer = document.querySelector('.business-footer');
        if (footer) {
            observer.observe(footer);
        }
    }

    function typewriterEffect(element, text, linkUrl) {
        let index = 0;
        element.textContent = '';

        const typeInterval = setInterval(() => {
            if (index < text.length) {
                element.textContent += text[index];
                index++;
            } else {
                clearInterval(typeInterval);
                // Convert to clickable link after animation completes
                setTimeout(() => {
                    const link = document.createElement('a');
                    link.href = linkUrl;
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    link.textContent = text;
                    element.innerHTML = '';
                    element.appendChild(link);
                }, 300); // Small delay before converting to link
            }
        }, 100); // 100ms delay between each letter
    }

    // Animated statistics counter
    function initStatsCounter() {
        const statNumbers = document.querySelectorAll('.stat-number');
        let hasAnimated = false;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !hasAnimated) {
                    hasAnimated = true;
                    statNumbers.forEach(stat => {
                        const target = parseInt(stat.getAttribute('data-target'));
                        animateCounter(stat, target);
                    });
                    observer.disconnect();
                }
            });
        }, {
            threshold: 0.5
        });

        const footer = document.querySelector('.business-footer');
        if (footer) {
            observer.observe(footer);
        }
    }

    function animateCounter(element, target) {
        let current = 0;
        const increment = target / 50; // 50 steps for smooth animation
        const duration = 2000; // 2 seconds
        const stepTime = duration / 50;

        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            element.textContent = Math.floor(current);
        }, stepTime);
    }

    // Smooth scroll for anchor links
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                const href = this.getAttribute('href');
                if (href !== '#' && href.length > 1) {
                    e.preventDefault();
                    const target = document.querySelector(href);
                    if (target) {
                        target.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                }
            });
        });
    }

    // Initialize all footer features
    initCodenesthubTypewriter();
    initStatsCounter();
    initSmoothScroll();
});
