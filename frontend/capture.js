function runCapture() {
  const name = document.getElementById("name").value.trim();
  if (!name) {
    alert("Please enter your name first.");
    return;
  }

  fetch(`/run_capture/${encodeURIComponent(name)}`)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      document.getElementById("response").textContent = data.output;
    })
    .catch((error) => {
      document.getElementById("response").textContent = `❌ Error: ${error.message}`;
    });
}
