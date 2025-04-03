document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("upload-form");
  
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
  
      const fileInput = document.getElementById("file-input");
      const file = fileInput.files[0];
      if (!file) {
        alert("Please upload a CSV file");
        return;
      }
  
      const formData = new FormData();
      formData.append("file", file);
  
      try {
        const response = await fetch("/upload", {
          method: "POST",
          body: formData
        });
  
        const data = await response.json();
  
        if (data.error) {
          alert("Error: " + data.error);
          return;
        }

        console.log("Server response:", data);
  
        fillTable("loo-positive", data.loo.positive);
        fillTable("loo-negative", data.loo.negative);
        fillTable("shap-influential", data.shapely.top_5_influential);
  
      } catch (err) {
        alert("Something went wrong: " + err.message);
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
  });
  