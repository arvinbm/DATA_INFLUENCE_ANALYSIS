document.addEventListener("DOMContentLoaded", () => {
  const form      = document.getElementById("upload-form");
  const fileInput = document.getElementById("file-input");
  const labelText = document.getElementById("file-label-text");
  const submitBtn = document.getElementById("submit-btn");
  const loading   = document.getElementById("loading");
  const errorBox  = document.getElementById("error-box");
  const results   = document.getElementById("results");

  // Update label text when a file is chosen
  fileInput.addEventListener("change", () => {
    labelText.textContent = fileInput.files[0]
      ? fileInput.files[0].name
      : "Choose a CSV file…";
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
      showError("Please select a CSV file before analyzing.");
      return;
    }

    // Reset UI
    hideError();
    results.classList.add("hidden");
    loading.classList.remove("hidden");
    submitBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.ERROR) {
        showError("Error: " + data.ERROR);
        return;
      }

      fillTable("loo-positive",    data.loo.positive);
      fillTable("loo-negative",    data.loo.negative);
      fillTable("shap-influential", data.shapely.top_5_influential);

      results.classList.remove("hidden");

    } catch (err) {
      showError("Something went wrong: " + err.message);
    } finally {
      loading.classList.add("hidden");
      submitBtn.disabled = false;
    }
  });

  function fillTable(tableId, rows) {
    const tbody = document.getElementById(tableId).querySelector("tbody");
    tbody.innerHTML = "";
    for (const row of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${row.index}</td><td>${row.value}</td>`;
      tbody.appendChild(tr);
    }
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove("hidden");
  }

  function hideError() {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
  }
});
