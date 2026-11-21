const form = document.getElementById("predictionForm");
const resultCard = document.getElementById("result");
const errorMessage = document.getElementById("errorMessage");
const submitButton = document.getElementById("submitButton");

function optionalNumber(value) {
  return value.trim() === "" ? null : Number(value);
}
function optionalBoolean(value) {
  if (value === "true") return true;
  if (value === "false") return false;
  return null;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultCard.classList.add("hidden");
  errorMessage.classList.add("hidden");
  submitButton.disabled = true;
  submitButton.textContent = "Processing...";

  const data = new FormData(form);
  const payload = {
    age: Number(data.get("age")),
    sex: data.get("sex"),
    cp: data.get("cp"),
    trestbps: optionalNumber(data.get("trestbps")),
    chol: optionalNumber(data.get("chol")),
    fbs: optionalBoolean(data.get("fbs")),
    restecg: data.get("restecg") || null,
    thalch: optionalNumber(data.get("thalch")),
    exang: optionalBoolean(data.get("exang")),
    oldpeak: optionalNumber(data.get("oldpeak")),
    slope: data.get("slope") || null,
    ca: optionalNumber(data.get("ca")),
    thal: data.get("thal") || null
  };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Prediction request failed.");

    document.getElementById("resultTitle").textContent =
      body.prediction === 1 ? "Heart Disease Pattern Detected" : "No Heart Disease Pattern Detected";
    document.getElementById("resultText").textContent = body.result;
    document.getElementById("probability").textContent = `${body.model_probability_percent}%`;
    document.getElementById("notice").textContent = body.notice;
    resultCard.classList.remove("hidden");
    resultCard.scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.classList.remove("hidden");
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = 'Predict Result <span>→</span>';
  }
});

form.addEventListener("reset", () => {
  resultCard.classList.add("hidden");
  errorMessage.classList.add("hidden");
});
