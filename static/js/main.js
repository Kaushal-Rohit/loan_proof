/* main.js – Loan Predictor frontend */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById("submit-btn");
    const btnText   = document.getElementById("btn-text");
    const spinner   = document.getElementById("btn-spinner");
    const errorBox  = document.getElementById("error-box");
    const resultCard = document.getElementById("result-card");

    // Reset
    errorBox.classList.add("d-none");
    errorBox.innerHTML = "";
    resultCard.classList.add("d-none");

    // Show loading
    submitBtn.disabled = true;
    spinner.classList.remove("d-none");
    btnText.textContent = "Predicting…";

    try {
      const formData = new FormData(form);

      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!data.success) {
        // Build error list safely using DOM methods to prevent XSS
        errorBox.innerHTML = "";
        const strong = document.createElement("strong");
        strong.textContent = "Please fix the following errors:";
        const ul = document.createElement("ul");
        ul.className = "mb-0 mt-1";
        data.errors.forEach((msg) => {
          const li = document.createElement("li");
          li.textContent = msg;
          ul.appendChild(li);
        });
        errorBox.appendChild(strong);
        errorBox.appendChild(ul);
        errorBox.classList.remove("d-none");
        return;
      }

      // Display result
      const icon  = document.getElementById("result-icon");
      const title = document.getElementById("result-title");
      const text  = document.getElementById("result-text");
      const bar   = document.getElementById("result-bar");
      const label = document.getElementById("result-prob-label");

      if (data.approved) {
        resultCard.classList.add("approved-card");
        resultCard.classList.remove("rejected-card");
        icon.textContent  = "✅";
        title.textContent = "Loan Approved!";
        title.className   = "fw-bold text-success";
        text.textContent  = "Congratulations! Based on your profile, you are likely eligible for this loan.";
        bar.className     = "progress-bar bg-success";
      } else {
        resultCard.classList.add("rejected-card");
        resultCard.classList.remove("approved-card");
        icon.textContent  = "❌";
        title.textContent = "Loan Not Approved";
        title.className   = "fw-bold text-danger";
        text.textContent  = "Based on your profile, this loan application is unlikely to be approved. Consider improving your credit score or reducing your DTI ratio.";
        bar.className     = "progress-bar bg-danger";
      }

      // Animate bar
      setTimeout(() => {
        bar.style.width = data.probability + "%";
        label.textContent = `Model confidence: ${data.probability}%`;
      }, 100);

      resultCard.classList.remove("d-none");
      resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      errorBox.innerHTML = "An unexpected error occurred. Please try again.";
      errorBox.classList.remove("d-none");
    } finally {
      submitBtn.disabled = false;
      spinner.classList.add("d-none");
      btnText.textContent = "🔍 Predict Eligibility";
    }
  });
});
