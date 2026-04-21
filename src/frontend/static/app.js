const resultNode = document.getElementById("result");
const insightsNode = document.getElementById("insights-summary");
const comparisonNode = document.getElementById("model-comparison");
const comparisonGraphsNode = document.getElementById("comparison-graphs");
const form = document.getElementById("prediction-form");
const submitButton = form.querySelector('button[type="submit"]');

function metricBlock(label, value) {
  return `<div><div class="muted">${label}</div><div class="metric">${value}</div></div>`;
}

async function loadInsights() {
  const response = await fetch("/insights");
  const data = await response.json();

  insightsNode.innerHTML = `
    <p><strong>Best post type for likes:</strong> ${data.best_for_likes}</p>
    <p><strong>Best post type for comments:</strong> ${data.best_for_comments}</p>
    <ul class="insight-list">
      ${data.post_type_averages
        .map(
          (item) =>
            `<li>${item.post_type}: ${Math.round(item.likes)} avg likes, ${Math.round(item.comments)} avg comments</li>`
        )
        .join("")}
    </ul>
  `;

  const likesMetrics = Object.entries(data.model_comparison.likes)
    .map(([name, metrics]) => `<li>${name}: MAE ${metrics.mae}, R² ${metrics.r2}</li>`)
    .join("");
  const commentsMetrics = Object.entries(data.model_comparison.comments)
    .map(([name, metrics]) => `<li>${name}: MAE ${metrics.mae}, R² ${metrics.r2}</li>`)
    .join("");

  comparisonNode.innerHTML = `
    <p><strong>Selected likes model:</strong> ${data.model_comparison.selected_models.likes}</p>
    <ul class="metric-list">${likesMetrics}</ul>
    <p><strong>Selected comments model:</strong> ${data.model_comparison.selected_models.comments}</p>
    <ul class="metric-list">${commentsMetrics}</ul>
  `;

  comparisonGraphsNode.innerHTML = `
    <div class="chart-stack">
      <img src="${data.model_comparison.charts.mae}" alt="Model comparison MAE chart" class="chart-image">
      <img src="${data.model_comparison.charts.r2}" alt="Model comparison R2 chart" class="chart-image">
    </div>
  `;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    followers: Number(document.getElementById("followers").value),
    post_type: document.getElementById("post_type").value,
    caption_length: Number(document.getElementById("caption_length").value),
    hashtags_count: Number(document.getElementById("hashtags_count").value),
    hour_posted: Number(document.getElementById("hour_posted").value),
    is_weekend: document.getElementById("is_weekend").checked,
    sponsorship: document.getElementById("sponsorship").checked,
  };

  submitButton.disabled = true;
  submitButton.textContent = "Predicting...";
  resultNode.innerHTML = `<p class="muted">Running prediction...</p>`;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || "Prediction request failed.");
    }

    const data = await response.json();

    resultNode.innerHTML = `
      <div class="result">
        ${metricBlock("Predicted Likes", data.predicted_likes)}
        ${metricBlock("Predicted Comments", data.predicted_comments)}
        <p>${data.recommendation}</p>
      </div>
    `;
  } catch (error) {
    resultNode.innerHTML = `
      <div class="result">
        <p class="error-text">Prediction failed. Restart the server after training and try again.</p>
        <p class="muted">${error.message}</p>
      </div>
    `;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predict engagement";
  }
});

loadInsights().catch(() => {
  insightsNode.textContent = "Run the training command first to generate project insights.";
  comparisonNode.textContent = "Model metrics will appear after training finishes.";
  comparisonGraphsNode.textContent = "Run the training command to generate graph-based model comparison.";
});
