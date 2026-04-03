document.addEventListener("DOMContentLoaded", () => {

    // ==========================================
    // A. FETCH REAL DATA
    // ==========================================
    const storedData = localStorage.getItem('fintwin_data');

    let aiData = { status: "approved", odds: 87, main_issue: "None", coach_advice: "Borrower demonstrates strong financial stability with manageable debt obligations. Recommended for approval." };
    let userData = { income: 850000, loan: 2500000, cibil: 750, assets: 7000000 };

    if (storedData) {
        const parsed = JSON.parse(storedData);
        aiData = parsed.ai_result;
        userData = parsed.user_inputs;
    }

    const odds = Math.round(aiData.odds);
    const riskScore = 100 - odds;
    const isApproved = aiData.status === "approved";
    const mainColor = isApproved ? '#10B981' : '#EF4444';

    // ==========================================
    // B. IMMEDIATELY POPULATE ALL VALUES (no skeleton delay)
    // ==========================================
    const formatMoney = (num) => '₹' + num.toLocaleString('en-IN');

    const decisionBadge = document.getElementById('decisionBadge');
    if (decisionBadge) {
        decisionBadge.innerText = aiData.status.toUpperCase();
        decisionBadge.style.color = mainColor;
        decisionBadge.style.backgroundColor = isApproved ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)';
    }

    // Handle missing bank_offers (e.g. from older localStorage caches)
    let bankOffersData = aiData.bank_offers;
    if (isApproved && (!bankOffersData || bankOffersData.length === 0)) {
        if (odds >= 85) {
            bankOffersData = [
                {"name": "State Bank of India", "interest": "8.25%", "type": "Public Sector", "icon": "building-columns"},
                {"name": "HDFC Bank", "interest": "8.50%", "type": "Private Sector", "icon": "landmark"},
                {"name": "ICICI Bank", "interest": "8.65%", "type": "Private Sector", "icon": "building"},
                {"name": "Bank of Baroda", "interest": "8.40%", "type": "Public Sector", "icon": "university"}
            ];
        } else if (odds >= 70) {
            bankOffersData = [
                {"name": "Axis Bank", "interest": "9.25%", "type": "Private Sector", "icon": "building"},
                {"name": "Kotak Mahindra Bank", "interest": "9.50%", "type": "Private Sector", "icon": "landmark"},
                {"name": "Punjab National Bank", "interest": "8.90%", "type": "Public Sector", "icon": "building-columns"}
            ];
        } else {
             bankOffersData = [
                {"name": "NBFC / Private Lenders", "interest": "11.5%", "type": "NBFC", "icon": "hand-holding-dollar"},
                {"name": "Bajaj Finserv", "interest": "12.0%", "type": "NBFC", "icon": "coins"}
            ];
        }
    }

    // Render Bank Offers
    const bankSection = document.getElementById('bankOffersSection');
    const bankGrid = document.getElementById('bankOffersGrid');

    if (isApproved && bankOffersData && bankOffersData.length > 0) {
        if (bankSection) bankSection.style.display = 'block';
        if (bankGrid) {
            bankGrid.innerHTML = '';
            bankOffersData.forEach((bank, index) => {
                const delay = index * 0.1;
                bankGrid.innerHTML += `
                    <div class="bank-card" style="animation-delay: ${delay}s">
                        <div class="bank-icon-container">
                            <i class="fa-solid fa-${bank.icon || 'building-columns'}"></i>
                        </div>
                        <div class="bank-details">
                            <h4>${bank.name}</h4>
                            <span class="bank-type">${bank.type || 'Bank'}</span>
                        </div>
                        <div class="bank-rate">
                            <span class="rate-label">Interest Rate</span>
                            <span class="rate-value">${bank.interest}</span>
                        </div>
                    </div>
                `;
            });
        }
    }

    // Simple counter animation — batched, non-blocking
    function animateCounter(el, end, duration, formatFn) {
        if (!el) return;
        let start = 0;
        let startTime = null;
        function tick(now) {
            if (!startTime) startTime = now;
            const progress = Math.min((now - startTime) / duration, 1);
            const value = Math.floor(progress * (end - start) + start);
            el.innerHTML = formatFn(value);
            if (progress < 1) requestAnimationFrame(tick);
            else el.innerHTML = formatFn(end);
        }
        requestAnimationFrame(tick);
    }

    // Fire all counters at once — they share the same rAF timing internally
    animateCounter(
        document.getElementById('riskScoreValue'), riskScore, 1200,
        (v) => `${v}<span style="font-size:14px;color:#64748b;font-weight:500">/100</span>`
    );
    animateCounter(document.getElementById('aiConfidenceValue'), odds, 1200, (v) => `${v}%`);
    animateCounter(document.getElementById('gaugeValueText'), odds, 1200, (v) => `${v}%`);
    animateCounter(document.getElementById('displayIncome'), userData.income, 1500, formatMoney);
    animateCounter(document.getElementById('displayLoan'), userData.loan, 1500, formatMoney);
    animateCounter(document.getElementById('displayCredit'), userData.cibil, 1500, (v) => v);
    animateCounter(document.getElementById('displayAssets'), userData.assets || 7000000, 1500, formatMoney);

    const monthlyPay = Math.round((userData.loan * 1.10) / 36);
    animateCounter(document.getElementById('estPaymentValue'), monthlyPay, 1500, formatMoney);

    // ==========================================
    // C. FAST ENTRANCE ANIMATIONS — stagger via CSS only
    // ==========================================
    const animatedElements = document.querySelectorAll('.animate-in');
    animatedElements.forEach((el, i) => {
        // Cap the max delay to 0.4s so nothing stays invisible
        el.style.animationDelay = `${Math.min(i * 0.05, 0.4)}s`;
    });

    // ==========================================
    // D. PROGRESS BARS — trigger immediately
    // ==========================================
    const progressBars = document.querySelectorAll('.progress-fill');
    // Use a small timeout so the CSS transition can run
    setTimeout(() => {
        progressBars.forEach(bar => {
            const w = bar.getAttribute('data-width');
            if (w) bar.style.width = w;
        });
    }, 300);

    // ==========================================
    // E. CHARTS — render immediately, no delay
    // ==========================================
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.color = '#64748b';

    const isDark = document.body.classList.contains('dark-theme');
    const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)';

    const premiumTooltip = {
        backgroundColor: isDark ? 'rgba(15,23,42,0.95)' : 'rgba(15,23,42,0.9)',
        titleFont: { weight: 700, size: 13 },
        bodyFont: { size: 12 },
        padding: 12,
        cornerRadius: 10,
        borderColor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)',
        borderWidth: 1,
        displayColors: true,
        boxPadding: 4,
    };

    // 1. Gauge Chart
    const ctxProb = document.getElementById('probabilityChart')?.getContext('2d');
    if (ctxProb) {
        const gaugeNeedle = {
            id: 'gaugeNeedle',
            afterDatasetDraw(chart) {
                const { ctx, data } = chart;
                ctx.save();
                const meta = chart.getDatasetMeta(0).data[0];
                const xCenter = meta.x;
                const yCenter = meta.y;
                const outerRadius = meta.outerRadius;
                const score = data.datasets[0].data[0];
                const angle = Math.PI + (score / 100) * Math.PI;

                ctx.translate(xCenter, yCenter);
                ctx.rotate(angle);
                ctx.beginPath();
                ctx.moveTo(0, -2.5);
                ctx.lineTo(outerRadius - 12, 0);
                ctx.lineTo(0, 2.5);
                ctx.fillStyle = isDark ? '#e2e8f0' : '#1e293b';
                ctx.fill();
                ctx.rotate(-angle);
                ctx.beginPath();
                ctx.arc(0, 0, 7, 0, Math.PI * 2);
                ctx.fillStyle = isDark ? '#e2e8f0' : '#1e293b';
                ctx.fill();
                ctx.beginPath();
                ctx.arc(0, 0, 3, 0, Math.PI * 2);
                ctx.fillStyle = '#22c55e';
                ctx.fill();
                ctx.restore();
            }
        };

        const gradientProb = ctxProb.createLinearGradient(0, 0, 300, 0);
        gradientProb.addColorStop(0, '#EF4444');
        gradientProb.addColorStop(0.35, '#F59E0B');
        gradientProb.addColorStop(0.65, '#10B981');
        gradientProb.addColorStop(1, '#22C55E');

        new Chart(ctxProb, {
            type: 'doughnut',
            data: {
                labels: ['Probability', 'Remaining'],
                datasets: [{
                    data: [odds, 100 - odds],
                    backgroundColor: [gradientProb, isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)'],
                    borderWidth: 0,
                    circumference: 180,
                    rotation: 270,
                    cutout: '86%',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2,
                layout: { padding: { top: 0, bottom: 0 } },
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                animation: { duration: 1000, easing: 'easeOutQuart' }
            },
            plugins: [gaugeNeedle]
        });
    }

    // 2. Trend Line Chart
    const ctxTrend = document.getElementById('trendChart')?.getContext('2d');
    if (ctxTrend) {
        const trendGradient = ctxTrend.createLinearGradient(0, 0, 0, 300);
        trendGradient.addColorStop(0, 'rgba(16, 185, 129, 0.12)');
        trendGradient.addColorStop(1, 'rgba(16, 185, 129, 0)');

        new Chart(ctxTrend, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                datasets: [{
                    label: 'Risk Score',
                    data: [15, 22, 23, 18, 12, 6, 5, 8, 18, 22, 28, riskScore],
                    borderColor: '#10B981',
                    backgroundColor: trendGradient,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#10B981',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'bottom', labels: { usePointStyle: true, boxWidth: 8, padding: 16 } },
                    tooltip: premiumTooltip
                },
                scales: {
                    y: { min: 0, max: 100, ticks: { stepSize: 25 }, border: { display: false }, grid: { color: gridColor } },
                    x: { grid: { display: false }, border: { display: false } }
                },
                interaction: { intersect: false, mode: 'index' },
                animation: { duration: 1200, easing: 'easeOutQuart' }
            }
        });
    }

    // 3. Asset Doughnut
    const ctxAsset = document.getElementById('assetChart')?.getContext('2d');
    if (ctxAsset) {
        new Chart(ctxAsset, {
            type: 'doughnut',
            data: {
                labels: ['Residential', 'Commercial', 'Luxury', 'Bank'],
                datasets: [{
                    data: [45, 20, 10, 25],
                    backgroundColor: ['#10B981', '#3b82f6', '#f59e0b', '#8b5cf6'],
                    borderWidth: 3,
                    borderColor: isDark ? '#1e293b' : '#ffffff',
                    hoverOffset: 6,
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false, cutout: '72%',
                plugins: {
                    legend: { position: 'bottom', labels: { usePointStyle: true, padding: 14, boxWidth: 8 } },
                    tooltip: premiumTooltip
                },
                animation: { duration: 1200, easing: 'easeOutQuart' }
            }
        });
    }

    // 4. Financial Risk Metrics Bar
    const cvsMetrics = document.getElementById('metricsChart')?.getContext('2d');
    if (cvsMetrics) {
        const metricsGrad = cvsMetrics.createLinearGradient(0, 0, 0, 250);
        metricsGrad.addColorStop(0, '#34d399');
        metricsGrad.addColorStop(1, 'rgba(52,211,153,0.3)');

        new Chart(cvsMetrics, {
            type: 'bar',
            data: {
                labels: ['LTI Ratio', 'Asset Coverage', 'Credit Score', 'Risk Score'],
                datasets: [{ label: 'Score', data: [60, 72, 85, riskScore], backgroundColor: metricsGrad, borderRadius: 8, barThickness: 36, borderSkipped: false }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: premiumTooltip },
                scales: {
                    y: { beginAtZero: true, max: 100, grid: { color: gridColor }, border: { display: false } },
                    x: { grid: { display: false }, border: { display: false } }
                },
                animation: { duration: 1200, easing: 'easeOutQuart' }
            }
        });
    }

    // ==========================================
    // F. FEATURES
    // ==========================================

    // PDF Export
    const exportPdfBtn = document.getElementById('exportPdfBtn');
    if (exportPdfBtn) {
        exportPdfBtn.addEventListener('click', () => {
            const element = document.querySelector('.dashboard-container');
            const opt = {
                margin: 10,
                filename: 'SmartCredit_AI_Report.pdf',
                image: { type: 'jpeg', quality: 0.98 },
                html2canvas: { scale: 2, useCORS: true, letterRendering: true, backgroundColor: null },
                jsPDF: { unit: 'mm', format: 'a3', orientation: 'landscape' }
            };

            // Hide controls
            const hiddenEls = [];
            document.querySelectorAll('button, .navbar').forEach(btn => {
                hiddenEls.push({ el: btn, d: btn.style.display });
                btn.style.display = 'none';
            });

            // Remove animations
            const animEls = [];
            document.querySelectorAll('.animate-in').forEach(node => {
                animEls.push({ el: node, a: node.style.animation, o: node.style.opacity, t: node.style.transform });
                node.style.animation = 'none';
                node.style.opacity = '1';
                node.style.transform = 'none';
            });

            if (window.html2pdf) {
                html2pdf().set(opt).from(element).save().then(() => {
                    hiddenEls.forEach(i => { i.el.style.display = i.d; });
                    animEls.forEach(i => { i.el.style.animation = i.a; i.el.style.opacity = i.o; i.el.style.transform = i.t; });
                    if (window.showToast) window.showToast('PDF Exported Successfully!', 'success');
                });
            }
        });
    }

    // UI Tour
    const tourBtn = document.getElementById('tourBtn');
    if (tourBtn) {
        tourBtn.addEventListener('click', () => {
            if (window.introJs) {
                introJs().setOptions({
                    showProgress: true,
                    steps: [
                        { title: "Welcome", intro: "This is the AI Decision Dashboard." },
                        { element: document.querySelector('.decision-card'), intro: "The final AI lending decision and overall risk." },
                        { element: document.querySelector('.metrics-grid'), intro: "Key financial markers evaluated by the ML model." },
                        { element: document.querySelector('.charts-grid'), intro: "Chart breakdowns of probability and risk factors." }
                    ]
                }).start();
            }
        });
    }

    // Confetti — only for truly great scores
    if (odds >= 85 && window.confetti) {
        setTimeout(() => {
            confetti({ particleCount: 80, spread: 70, origin: { y: 0.6 }, colors: ['#22c55e', '#3b82f6', '#f59e0b'] });
        }, 2000);
    }

    // ==========================================
    // G. COMPARISON MODE
    // ==========================================
    const comparisonToggle = document.getElementById('comparisonToggle');
    const comparisonBar = document.getElementById('comparisonBar');
    const coachAdviceText = document.getElementById('coachAdviceText');

    const origValues = {
        income: document.getElementById('displayIncome')?.innerText,
        loan: document.getElementById('displayLoan')?.innerText,
        odds: odds,
        risk: riskScore
    };

    if (comparisonToggle) {
        comparisonToggle.addEventListener('change', (e) => {
            const isSim = e.target.checked;
            comparisonBar.style.display = isSim ? 'flex' : 'none';

            if (isSim) {
                const simOdds = Math.min(98, odds + 8);
                const simRisk = 100 - simOdds;
                updateUI(simOdds, simRisk, "₹950,000", "₹2,000,000");
                genStory(simOdds, true);
            } else {
                updateUI(origValues.odds, origValues.risk, origValues.income, origValues.loan);
                genStory(odds, false);
            }
        });
    }

    function updateUI(newOdds, newRisk, newIncome, newLoan) {
        const g = document.getElementById('gaugeValueText');
        const r = document.getElementById('riskScoreValue');
        const i = document.getElementById('displayIncome');
        const l = document.getElementById('displayLoan');
        if (g) g.innerText = newOdds + '%';
        if (r) r.innerHTML = `<strong>${newRisk}</strong><span class="muted">/100</span>`;
        if (i) i.innerText = newIncome;
        if (l) l.innerText = newLoan;

        const probChart = Chart.getChart("probabilityChart");
        if (probChart) {
            probChart.data.datasets[0].data = [newOdds, 100 - newOdds];
            probChart.update('none'); // 'none' = no animation for instant update
        }
    }

    function genStory(score, isComp) {
        let s = "";
        if (score > 80) s = `Your financial health is in the top 5%. ${isComp ? "In this scenario, your" : "Your"} high assets and stable credit score make you ideal for our lowest rates.`;
        else if (score > 60) s = `Moderate risk. Increasing collateral or lowering the loan term could shift you into the premium approval tier.`;
        else s = `Caution indicated. The AI detected high debt-to-income sensitivity. Try simulating lower loan amounts.`;
        if (coachAdviceText) coachAdviceText.innerText = s;
    }

    genStory(odds, false);
});