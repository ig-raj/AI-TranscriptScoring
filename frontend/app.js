// frontend/app.js

const scoreBtn = document.getElementById("scoreBtn");
const transcriptInput = document.getElementById("transcript");
const durationInput = document.getElementById("duration");

const loadingEl = document.getElementById("loading");
const errorEl = document.getElementById("error");
const resultsEl = document.getElementById("results");

const overallScoreEl = document.getElementById("overallScore");
const wordCountEl = document.getElementById("wordCount");
const sentenceCountEl = document.getElementById("sentenceCount");
const wpmEl = document.getElementById("wpm");

const criteriaTableEl = document.getElementById("criteriaTable");

scoreBtn.addEventListener("click", async () => {
  const transcript = transcriptInput.value.trim();
  let duration = parseFloat(durationInput.value);

  if (!transcript) {
    showError("Please paste a transcript before scoring.");
    return;
  }

  // Default duration = 52 seconds if not provided
  if (isNaN(duration) || duration <= 0) {
    duration = 52;
  }

  setLoading(true);
  clearError();
  hideResults();

  try {
    const response = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript, duration }),
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || "Something went wrong while scoring.");
      return;
    }

    renderResults(data);
  } catch (err) {
    console.error(err);
    showError("Could not connect to the server. Is the backend running?");
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  loadingEl.classList.toggle("hidden", !isLoading);
  scoreBtn.disabled = isLoading;
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function clearError() {
  errorEl.textContent = "";
  errorEl.classList.add("hidden");
}

function hideResults() {
  resultsEl.classList.add("hidden");
}

function renderResults(data) {
  resultsEl.classList.remove("hidden");

  overallScoreEl.textContent = data.overall_score.toFixed(1);
  wordCountEl.textContent = data.words;
  sentenceCountEl.textContent = data.sentences;
  wpmEl.textContent = data.wpm;

  criteriaTableEl.innerHTML = "";

  (data.criteria || []).forEach((c) => {
    const card = document.createElement("div");
    card.className = "criteria-card";

    // ----- Header: name + total score -----
    const header = document.createElement("div");
    header.className = "criteria-header";

    const nameEl = document.createElement("div");
    nameEl.className = "criteria-name";
    nameEl.textContent = c.name;

    const scoreEl = document.createElement("div");
    scoreEl.className = "criteria-score";
    scoreEl.textContent = `${c.score.toFixed(2)} / ${c.max}`;

    header.appendChild(nameEl);
    header.appendChild(scoreEl);
    card.appendChild(header);

    // ----- Sub-scores based on criterion name -----
    const subsContainer = document.createElement("div");
    subsContainer.className = "sub-scores";

    const d = c.details || {};

    if (c.name === "Content & Structure") {
      // 4/5, 24/30, 5/5 style
      addSubScore(
        subsContainer,
        "Salutation level",
        d.salutation_points ?? 0,
        5
      );
      const must = d.must_have_score ?? 0;
      const good = d.good_have_score ?? 0;
      addSubScore(
        subsContainer,
        "Keyword presence (must + good)",
        must + good,
        30
      );
      addSubScore(
        subsContainer,
        "Flow",
        d.flow_points ?? 0,
        5
      );
    } else if (c.name === "Speech Rate") {
      addSubScore(
        subsContainer,
        "Speech as WPM",
        d.rule_score ?? d.rule_score === 0 ? d.rule_score : d.rule_score,
        d.max_rule_score ?? 10
      );
    } else if (c.name === "Language & Grammar") {
      addSubScore(
        subsContainer,
        "Grammar errors (heuristic)",
        d.grammar_score ?? 0,
        10
      );
      addSubScore(
        subsContainer,
        "Vocabulary richness (TTR)",
        d.vocab_score ?? 0,
        10
      );
    } else if (c.name === "Clarity") {
      addSubScore(
        subsContainer,
        "Filler word rate",
        d.rule_score ?? 0,
        d.max_rule_score ?? 15
      );
    } else if (c.name === "Engagement") {
      addSubScore(
        subsContainer,
        "Sentiment positivity",
        d.rule_score ?? 0,
        d.max_rule_score ?? 15
      );
    }

    card.appendChild(subsContainer);

    // ----- Feedback text -----
    if (c.feedback) {
      const feedbackTitle = document.createElement("div");
      feedbackTitle.className = "criteria-feedback-label";
      feedbackTitle.textContent = "Feedback";

      const feedbackEl = document.createElement("div");
      feedbackEl.className = "criteria-feedback";
      feedbackEl.textContent = c.feedback;

      card.appendChild(feedbackTitle);
      card.appendChild(feedbackEl);
    }

    criteriaTableEl.appendChild(card);
  });
}

/**
 * Helper: add one "→ label score / max" row.
 */
function addSubScore(container, label, score, max) {
  const row = document.createElement("div");
  row.className = "sub-score-row";

  const labelEl = document.createElement("span");
  labelEl.className = "sub-score-label";
  labelEl.textContent = `→ ${label}`;

  const valueEl = document.createElement("span");
  valueEl.className = "sub-score-value";
  valueEl.textContent = `${Number(score).toFixed(2)} / ${max}`;

  row.appendChild(labelEl);
  row.appendChild(valueEl);
  container.appendChild(row);
}
