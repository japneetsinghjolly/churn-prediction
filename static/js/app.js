(() => {
  const form = document.getElementById('prediction-form');
  const predictButton = document.getElementById('predict-button');
  const resetButton = document.getElementById('reset-button');
  const emptyState = document.getElementById('empty-state');
  const resultState = document.getElementById('result-state');
  const loadingState = document.getElementById('loading-state');
  const errorMessage = document.getElementById('error-message');
  const probabilityRing = document.getElementById('probability-ring');
  const probabilityValue = document.getElementById('probability-value');
  const riskLabel = document.getElementById('risk-label');
  const resultSummary = document.getElementById('result-summary');
  const recommendationText = document.getElementById('recommendation-text');

  const $ = (id) => document.getElementById(id);
  const show = (element) => { element.hidden = false; };
  const hide = (element) => { element.hidden = true; };

  function clearMessages() {
    hide(errorMessage);
    errorMessage.textContent = '';
  }

  function setBusy(busy) {
    predictButton.disabled = busy;
    predictButton.innerHTML = busy
      ? '<span class="loader" style="width:15px;height:15px;border-width:2px"></span> Analyzing…'
      : '<span class="button-spark">✳</span> Analyze churn risk <span class="button-arrow">→</span>';
    if (busy) {
      hide(emptyState);
      hide(resultState);
      show(loadingState);
    } else {
      hide(loadingState);
    }
  }

  function getPayload() {
    return {
      tenure: Number($('tenure').value),
      MonthlyCharges: Number($('monthly').value),
      TotalCharges: Number($('total').value),
      InternetService: $('internet').value,
      Contract: $('contract').value,
      OnlineSecurity: $('security').value
    };
  }

  function displayResult(data) {
    const percent = Math.max(0, Math.min(100, Number(data.probability_percent)));
    probabilityValue.textContent = `${percent.toFixed(1)}%`;
    probabilityRing.style.setProperty('--prob', `${percent * 3.6}deg`);
    riskLabel.textContent = `${data.risk} risk`;
    riskLabel.className = `risk-label ${data.risk_key}`;
    resultSummary.textContent = data.risk === 'High'
      ? 'This customer has a higher estimated likelihood of leaving based on the selected details.'
      : data.risk === 'Medium'
        ? 'This customer has a moderate estimated likelihood of leaving. Monitor for changes.'
        : 'This customer has a lower estimated likelihood of leaving based on the selected details.';
    recommendationText.textContent = data.recommendation;
    probabilityRing.style.background = data.risk_key === 'high'
      ? `conic-gradient(from -90deg, #e45b61 ${percent * 3.6}deg, #f8e9e9 ${percent * 3.6}deg)`
      : data.risk_key === 'medium'
        ? `conic-gradient(from -90deg, #e6a33e ${percent * 3.6}deg, #f8f0e2 ${percent * 3.6}deg)`
        : `conic-gradient(from -90deg, #16b89b ${percent * 3.6}deg, #e6f5f1 ${percent * 3.6}deg)`;
    hide(emptyState);
    show(resultState);
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    clearMessages();
    if (!form.reportValidity()) return;
    setBusy(true);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(getPayload())
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
      displayResult(data);
    } catch (error) {
      show(errorMessage);
      errorMessage.textContent = error.message || 'Unable to reach the prediction service. Please try again.';
      show(emptyState);
    } finally {
      setBusy(false);
    }
  });

  resetButton.addEventListener('click', () => {
    form.reset();
    clearMessages();
    hide(resultState);
    hide(loadingState);
    show(emptyState);
    probabilityRing.style.setProperty('--prob', '0deg');
    probabilityRing.style.background = 'conic-gradient(from -90deg, #7068f5 0deg, #eeeff8 0deg)';
  });
})();
