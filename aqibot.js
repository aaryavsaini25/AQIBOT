async function enter(){
  const city = document.getElementById("cityInput").value;
  const year = document.getElementById("yearInput").value;

  const response = await fetch("/submit", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ city, year })
  });

  const result = await response.json();

  document.getElementById("result").innerHTML = result.message;

  const img = document.getElementById("result-image");
  if (result.image) {
    img.src = "data:image/png;base64," + result.image;
    img.style.display = "block";
  } else {
    img.style.display = "none";
  }
}