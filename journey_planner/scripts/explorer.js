let data = {};
let monthSelect = document.getElementById('monthSelect');
let destSelect = document.getElementById('destinationSelect');
let detailsDiv = document.getElementById('detailsTabs');

fetch('data/plans.json') // Converted from YAML to JSON
  .then(res => res.json())
  .then(json => {
    data = json;
    populateDropdowns();
  });

function populateDropdowns() {
  const destinations = [...new Set(data.map(plan => plan.start_point))];
  destSelect.innerHTML = destinations.map(d => `<option>${d}</option>`).join('');
  updateMonths();
  destSelect.addEventListener('change', updateMonths);
  monthSelect.addEventListener('change', updateTabs);
}

function updateMonths() {
  const selected = destSelect.value;
  const months = [...new Set(data.filter(p => p.start_point === selected).map(p => p.month))];
  monthSelect.innerHTML = months.map(m => `<option>${m}</option>`).join('');
  updateTabs();
}

function updateTabs() {
  const dest = destSelect.value;
  const month = monthSelect.value;
  const plan = data.find(p => p.start_point === dest && p.month === month);
  if (!plan) return (detailsDiv.innerHTML = 'No data available.');
  detailsDiv.innerHTML = plan.days.map(day => `
    <div class='day-tab'>
      <h3>Day ${day.day}</h3>
      ${day.attractions.map(a => `
        <div class='attraction'>
          <h4>${a.name}</h4>
          <p><strong>Miles:</strong> ${a.num_miles}</p>
          <p><strong>Reach:</strong> ${a.how_to_reach}</p>
          <p><strong>Restrooms:</strong> ${a.public_restrooms}</p>
          <p><strong>Food:</strong> ${a.food_recommendation}</p>
          <p><strong>Weather:</strong> ${a.weather_conditions}</p>
        </div>
      `).join('')}
    </div>
  `).join('');
}
