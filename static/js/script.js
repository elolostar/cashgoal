// Variables globales
let currentMatches = [];

// ============================================
// FONCTIONS PRINCIPALES
// ============================================

// Charger les matchs quand on change de compétition
$("#competition").change(function () {
  const competition = $(this).val();
  if (competition) {
    loadMatches(competition);
  }
});

// Charger les matchs
function loadMatches(competition) {
  const daysAhead = $("#period").val();

  $("#matchesList").html(
    '<div class="loader"></div><p class="text-center">Chargement des matchs...</p>',
  );

  $.ajax({
    url: "/api/todays-matches",
    method: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      competition: competition,
      days_ahead: parseInt(daysAhead),
    }),
    success: function (response) {
      if (response.success) {
        displayMatches(response.matches);
        currentMatches = response.matches;
      } else {
        $("#matchesList").html(
          '<p class="text-danger text-center">' + response.error + "</p>",
        );
      }
    },
    error: function () {
      $("#matchesList").html(
        '<p class="text-danger text-center">Erreur de chargement</p>',
      );
    },
  });
}

// Afficher les matchs
function displayMatches(matches) {
  if (matches.length === 0) {
    $("#matchesList").html(
      '<p class="text-warning text-center">Aucun match trouvé</p>',
    );
    return;
  }

  let html = '<div class="list-group">';
  matches.forEach((match) => {
    html += `
            <div class="list-group-item">
                <div class="row">
                    <div class="col-md-6">
                        <strong>${match.home_team} vs ${match.away_team}</strong>
                    </div>
                    <div class="col-md-3">
                        ${match.date} à ${match.time}
                    </div>
                    <div class="col-md-3">
                        <span class="badge bg-info">${match.competition}</span>
                    </div>
                </div>
            </div>
        `;
  });
  html += "</div>";

  $("#matchesList").html(html);
}

// Obtenir les prédictions
function getPredictions() {
  const competition = $("#competition").val();
  const daysAhead = $("#period").val();
  const nMatches = $("#nMatches").val();

  if (!competition) {
    alert("Veuillez sélectionner une compétition");
    return;
  }

  $("#predictionsContainer").hide();
  $("#predictionsList").html(
    '<div class="loader"></div><p class="text-center">Analyse en cours...</p>',
  );
  $("#predictionsContainer").show();

  $.ajax({
    url: "/api/predict-sure-bets",
    method: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      competition: competition,
      days_ahead: parseInt(daysAhead),
      n_matches: parseInt(nMatches),
    }),
    success: function (response) {
      if (response.success) {
        displayPredictions(response.predictions);
      } else {
        $("#predictionsList").html(
          '<p class="text-danger text-center">' + response.error + "</p>",
        );
      }
    },
    error: function () {
      $("#predictionsList").html(
        '<p class="text-danger text-center">Erreur de prédiction</p>',
      );
    },
  });
}

// Afficher les prédictions
function displayPredictions(predictions) {
  if (predictions.length === 0) {
    $("#predictionsList").html(
      '<p class="text-warning text-center">Aucune prédiction avec confiance suffisante</p>',
    );
    return;
  }

  let html = "";

  predictions.forEach((pred, index) => {
    const template = document
      .getElementById("predictionTemplate")
      .content.cloneNode(true);

    // Confiance
    const confidenceCircle = template.querySelector(".confidence-circle");
    confidenceCircle.textContent = pred.confidence + "%";

    // Titre du match
    template.querySelector(".match-title").innerHTML =
      `${pred.prediction_emoji} ${pred.home_team} vs ${pred.away_team}`;

    // Date/heure
    template.querySelector(".match-datetime").innerHTML =
      `<i class="far fa-calendar"></i> ${pred.date} à ${pred.time}`;

    // Barres de progression
    template.getElementById("homeProgress").style.width =
      pred.probabilities.home + "%";
    template.getElementById("drawProgress").style.width =
      pred.probabilities.draw + "%";
    template.getElementById("awayProgress").style.width =
      pred.probabilities.away + "%";

    // Probabilités
    template.querySelector(".homeProb").textContent = pred.probabilities.home;
    template.querySelector(".drawProb").textContent = pred.probabilities.draw;
    template.querySelector(".awayProb").textContent = pred.probabilities.away;

    // Scores exacts
    let scoresHtml = '<small class="text-muted">Scores exacts:</small><br>';
    pred.exact_scores.forEach((score) => {
      scoresHtml += `<span class="score-chip">${score.score} (${score.probability}%)</span> `;
    });
    template.querySelector(".exact-scores").innerHTML = scoresHtml;

    // Joueurs clés
    if (pred.key_players.home.length > 0 || pred.key_players.away.length > 0) {
      let playersHtml = '<small class="text-muted">Joueurs clés:</small><br>';

      if (pred.key_players.home.length > 0) {
        playersHtml += `<span class="badge bg-danger">🔴 ${pred.home_team}</span><br>`;
        pred.key_players.home.forEach((player) => {
          playersHtml += `<small>• ${player}</small><br>`;
        });
      }

      if (pred.key_players.away.length > 0) {
        playersHtml += `<span class="badge bg-primary">🔵 ${pred.away_team}</span><br>`;
        pred.key_players.away.forEach((player) => {
          playersHtml += `<small>• ${player}</small><br>`;
        });
      }

      template.querySelector(".key-players").innerHTML = playersHtml;
    }

    // Ajouter la prédiction à la liste
    const tempDiv = document.createElement("div");
    tempDiv.appendChild(template);
    html += tempDiv.innerHTML;
  });

  $("#predictionsList").html(html);

  // Animation
  $(".prediction-card").each(function (index) {
    $(this).css("animation-delay", index * 0.2 + "s");
  });
}

// Rechercher une équipe
function searchTeam() {
  const teamName = $("#teamSearch").val();

  if (!teamName) {
    alert("Veuillez entrer un nom d'équipe");
    return;
  }

  $("#teamResults").html(
    '<div class="loader"></div><p class="text-center">Recherche...</p>',
  );

  $.ajax({
    url: "/api/search-team",
    method: "POST",
    contentType: "application/json",
    data: JSON.stringify({ team_name: teamName }),
    success: function (response) {
      if (response.success) {
        displayTeamMatches(response.matches, response.team);
      } else {
        $("#teamResults").html(
          '<p class="text-danger text-center">' + response.error + "</p>",
        );
      }
    },
  });
}

// Afficher les matchs d'une équipe
function displayTeamMatches(matches, teamName) {
  if (matches.length === 0) {
    $("#teamResults").html(
      '<p class="text-warning text-center">Aucun match trouvé pour ' +
        teamName +
        "</p>",
    );
    return;
  }

  let html = `<h5>Matchs de ${teamName}:</h5>`;
  html += '<div class="list-group">';

  matches.forEach((match) => {
    html += `
            <button class="list-group-item list-group-item-action" onclick="analyzeMatch(${match.id})">
                <div class="row">
                    <div class="col-md-8">
                        ${match.home_team} vs ${match.away_team}
                    </div>
                    <div class="col-md-4">
                        ${match.date} - ${match.competition}
                    </div>
                </div>
            </button>
        `;
  });

  html += "</div>";
  $("#teamResults").html(html);
}

// Analyser un match spécifique
function analyzeMatch(matchId) {
  // Rediriger vers la page d'analyse
  window.location.href = "/match-analysis?id=" + matchId;
}

// Initialisation
$(document).ready(function () {
  console.log("Application prête !");

  // Tooltips Bootstrap
  var tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]'),
  );
  var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });
});
