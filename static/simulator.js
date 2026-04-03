document.addEventListener("DOMContentLoaded", () => {
    
    // ==========================================
    // 1. LOAD ORIGINAL DATA FROM MEMORY
    // ==========================================
    const storedData = localStorage.getItem('fintwin_data');
    
    let baseData = {
        income: 850000, 
        loan: 2500000, 
        cibil: 750,
        original_odds: 87,
        missed_payments: 0
    };

    if (storedData) {
        const parsed = JSON.parse(storedData);
        baseData.income = parsed.user_inputs.income || baseData.income;
        baseData.loan = parsed.user_inputs.loan || baseData.loan;
        baseData.cibil = parsed.user_inputs.cibil || baseData.cibil;
        baseData.original_odds = Math.round(parsed.ai_result.odds) || baseData.original_odds;
        
        if (baseData.cibil < 650) baseData.missed_payments = 2;
        else if (baseData.cibil < 720) baseData.missed_payments = 1;
    }

    const originalRisk = 100 - baseData.original_odds;

    // ==========================================
    // 2. SETUP UI ELEMENTS
    // ==========================================
    const incomeSlider = document.getElementById('slide-income');
    const loanSlider = document.getElementById('slide-loan');
    const durationSlider = document.getElementById('slide-duration');
    const rateSlider = document.getElementById('slide-rate');

    const incomeVal = document.getElementById('val-income');
    const loanVal = document.getElementById('val-loan');
    const durationVal = document.getElementById('val-duration');
    const rateVal = document.getElementById('val-rate');

    const simMonthlyPayment = document.getElementById('res-payment');
    const simRiskScore = document.getElementById('res-score');
    const simApproveProb = document.getElementById('res-prob');

    const origRiskVal = document.getElementById('origRiskVal');
    const origRiskStatus = document.getElementById('origRiskStatus');
    const newRiskVal = document.getElementById('sim-risk-value');
    const newRiskStatus = document.getElementById('sim-status-badge');
    const simWarningBox = document.getElementById('status-message');

    if(origRiskVal) origRiskVal.innerText = originalRisk;
    if(origRiskStatus) {
        origRiskStatus.innerText = originalRisk > 50 ? "High Risk" : "Approved";
        origRiskStatus.style.color = originalRisk > 50 ? "#EF4444" : "#10B981";
    }

    if(incomeSlider) incomeSlider.value = baseData.income;
    if(loanSlider) loanSlider.value = baseData.loan;

    const formatMoney = (num) => '₹' + parseInt(num).toLocaleString('en-IN');

    // ==========================================
    // 3. CHART SETUP WITH PREMIUM GRADIENTS
    // ==========================================
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.color = '#64748b';
    const ctx = document.getElementById('comparisonChart')?.getContext('2d');
    let compChart = null;

    if (ctx) {
        // Create gradient fills
        const riskGradient = ctx.createLinearGradient(0, 0, 0, 350);
        riskGradient.addColorStop(0, 'rgba(239, 68, 68, 0.85)');
        riskGradient.addColorStop(1, 'rgba(239, 68, 68, 0.4)');

        const approvalGradient = ctx.createLinearGradient(0, 0, 0, 350);
        approvalGradient.addColorStop(0, 'rgba(16, 185, 129, 0.85)');
        approvalGradient.addColorStop(1, 'rgba(16, 185, 129, 0.4)');

        compChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Original', 'Simulated'],
                datasets: [
                    {
                        label: 'Risk Score',
                        data: [originalRisk, originalRisk],
                        backgroundColor: riskGradient,
                        borderRadius: 8,
                        barThickness: 44,
                        borderSkipped: false,
                    },
                    {
                        label: 'Approval Probability',
                        data: [baseData.original_odds, baseData.original_odds],
                        backgroundColor: approvalGradient,
                        borderRadius: 8,
                        barThickness: 44,
                        borderSkipped: false,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: 0, max: 100,
                        grid: { color: 'rgba(0,0,0,0.04)', borderDash: [4, 4] },
                        border: { display: false },
                        ticks: { padding: 8 }
                    },
                    x: {
                        grid: { display: false },
                        border: { display: false },
                        ticks: { padding: 8 }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            boxWidth: 8,
                            padding: 20,
                            font: { weight: 500 }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15,23,42,0.9)',
                        titleFont: { weight: 700 },
                        bodyFont: { size: 13 },
                        padding: 12,
                        cornerRadius: 10,
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                    }
                },
                animation: {
                    duration: 800,
                    easing: 'easeOutQuart'
                }
            }
        });
    }

    // ==========================================
    // 4. THE SIMULATION ENGINE
    // ==========================================
    let prevRisk = originalRisk;

    function calculateEMI(principal, annualRate, months) {
        if (annualRate === 0) return principal / months;
        const r = (annualRate / 12) / 100;
        const emi = principal * r * (Math.pow(1 + r, months)) / (Math.pow(1 + r, months) - 1);
        return emi;
    }

    function animateValueChange(element, newValue, isFormatted = false) {
        if (!element) return;
        element.style.transition = 'transform 0.15s ease';
        element.style.transform = 'scale(0.95)';
        setTimeout(() => {
            element.innerText = isFormatted ? formatMoney(newValue) : newValue;
            element.style.transform = 'scale(1.05)';
            setTimeout(() => {
                element.style.transform = 'scale(1)';
            }, 100);
        }, 80);
    }

    function runSimulation() {
        const currentIncome = parseInt(incomeSlider?.value || baseData.income);
        const currentLoan = parseInt(loanSlider?.value || baseData.loan);
        const currentDuration = parseInt(durationSlider?.value || 36);
        const currentRate = parseFloat(rateSlider?.value || 8.0);

        // Update Slider Text with spring animation
        if(incomeVal) animateValueChange(incomeVal, currentIncome, true);
        if(loanVal) animateValueChange(loanVal, currentLoan, true);
        if(durationVal) {
            durationVal.style.transition = 'transform 0.15s ease';
            durationVal.innerText = `${currentDuration} mo`;
        }
        if(rateVal) {
            rateVal.style.transition = 'transform 0.15s ease';
            rateVal.innerText = `${currentRate.toFixed(1)}%`;
        }

        const emi = calculateEMI(currentLoan, currentRate, currentDuration);
        if(simMonthlyPayment) animateValueChange(simMonthlyPayment, emi, true);

        const annualDebt = emi * 12;
        const dti = annualDebt / currentIncome;
        const lti = currentLoan / currentIncome;
        
        let simulatedOdds = baseData.original_odds;
        
        if (lti > 3) simulatedOdds -= (lti - 3) * 5;
        if (lti < 2) simulatedOdds += (2 - lti) * 10;
        if (dti > 0.4) simulatedOdds -= (dti - 0.4) * 80;
        if (dti < 0.2) simulatedOdds += (0.2 - dti) * 50;

        simulatedOdds = Math.round(Math.max(5, Math.min(99, simulatedOdds)));
        const simulatedRisk = 100 - simulatedOdds;
        const isApproved = simulatedRisk <= 50;

        // Update Top Results
        if(simRiskScore) {
            simRiskScore.innerText = simulatedRisk;
            simRiskScore.style.color = isApproved ? 'var(--primary-navy)' : '#EF4444';
        }
        if(simApproveProb) {
            simApproveProb.innerText = `${simulatedOdds}%`;
            simApproveProb.style.color = isApproved ? '#10B981' : '#EF4444';
        }

        // Update Comparison
        if(newRiskVal) newRiskVal.innerText = simulatedRisk;
        if(newRiskStatus) {
            newRiskStatus.innerText = isApproved ? "Approved" : "High Risk";
            newRiskStatus.className = isApproved ? "badge approved" : "badge declined";
        }

        // Update Status Message with smooth transition
        if(simWarningBox) {
            simWarningBox.style.transition = 'all 0.4s ease';
            if (simulatedRisk > originalRisk && !isApproved) {
                simWarningBox.innerHTML = "⚠️ Warning: Simulated parameters significantly increase default risk.";
                simWarningBox.className = "status-message warning";
            } else if (simulatedRisk < originalRisk && isApproved) {
                simWarningBox.innerHTML = "✅ Success: These parameters improve your approval odds!";
                simWarningBox.className = "status-message success";
            } else {
                simWarningBox.innerHTML = "📊 Parameters adjusted. Profile remains functionally stable.";
                simWarningBox.className = "status-message neutral";
            }
        }

        prevRisk = simulatedRisk;

        // Update Chart
        requestAnimationFrame(() => {
            if(compChart) {
                compChart.data.datasets[0].data[1] = simulatedRisk;
                compChart.data.datasets[1].data[1] = simulatedOdds;
                compChart.update('none');
            }
        });
    }

    // ==========================================
    // 5. ATTACH EVENT LISTENERS
    // ==========================================
    const sliders = [incomeSlider, loanSlider, durationSlider, rateSlider];
    sliders.forEach(slider => {
        if (slider) {
            slider.addEventListener('input', runSimulation);
        }
    });

    // Initial run
    runSimulation();
});