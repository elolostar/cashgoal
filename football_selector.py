import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import json
from collections import defaultdict
import time
from scipy import stats
from scipy.stats import poisson, norm
import os
import pickle
warnings.filterwarnings('ignore')

class FootballMatchSelectorUltra:
    """
    Programme ULTRA-INTELLIGENT de prédiction de matchs de football
    Version ULTIME avec Lineups, Head2Head, Météo, Motivation et Contexte
    """
    
    def __init__(self, api_key="1b9fa9eead33409cb75f3d0a2df60324"):
        """
        Initialisation avec la clé API football-data.org
        """
        self.api_key = api_key
        self.base_url = "https://api.football-data.org/v4/"
        self.headers = {'X-Auth-Token': api_key}
        
        # Initialisation des modèles
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        
        # Compétitions disponibles
        self.competitions = {
            'PL': {'name': 'Premier League (Angleterre)', 'country': 'Angleterre', 'code': 'PL'},
            'PD': {'name': 'La Liga (Espagne)', 'country': 'Espagne', 'code': 'PD'},
            'SA': {'name': 'Serie A (Italie)', 'country': 'Italie', 'code': 'SA'},
            'BL1': {'name': 'Bundesliga (Allemagne)', 'country': 'Allemagne', 'code': 'BL1'},
            'FL1': {'name': 'Ligue 1 (France)', 'country': 'France', 'code': 'FL1'},
            'CL': {'name': 'UEFA Champions League', 'country': 'Europe', 'code': 'CL'},
            'EL': {'name': 'UEFA Europa League', 'country': 'Europe', 'code': 'EL'},
            'EC': {'name': 'UEFA Europa Conference League', 'country': 'Europe', 'code': 'EC'},
        }
        
        # Cache
        self.teams_cache = {}
        self.matches_cache = {}
        self.players_cache = {}
        self.scorers_cache = {}
        self.historical_results = {}
        self.lineups_cache = {}
        self.head2head_cache = {}
        self.standings_cache = {}  # NOUVEAU: Cache pour les classements
        self.weather_cache = {}     # NOUVEAU: Cache pour la météo
        self.team_schedule_cache = {} # NOUVEAU: Cache pour le calendrier des équipes
        
        # Paramètres pour le modèle de Poisson
        self.league_averages = self._init_league_averages()
        
        # Forces des équipes
        self.team_attack_strength = {}
        self.team_defense_strength = {}
        self.team_form = {}
        
        # Statistiques historiques
        self.historical_goals_distribution = self._init_historical_distributions()
        
        # Compteur de requêtes
        self.request_count = 0
        self.last_request_time = time.time()
        
        # Seuil de confiance
        self.MIN_CONFIDENCE_THRESHOLD = 0.50
        
        # Charger les modèles pré-entraînés
        self._load_trained_models()
    
    # ============================================
    # CHARGEMENT/SAUVEGARDE DES MODÈLES
    # ============================================
    
    def _load_trained_models(self):
        """Charge les modèles pré-entraînés s'ils existent"""
        try:
            if os.path.exists('models.pkl'):
                with open('models.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.models = data['models']
                    self.scaler = data['scaler']
                    self.is_trained = True
                    print("✅ Modèles pré-entraînés chargés!")
        except:
            pass
    
    def _save_trained_models(self):
        """Sauvegarde les modèles entraînés"""
        try:
            with open('models.pkl', 'wb') as f:
                pickle.dump({
                    'models': self.models,
                    'scaler': self.scaler
                }, f)
            print("✅ Modèles sauvegardés!")
        except:
            pass
    
    # ============================================
    # INITIALISATION
    # ============================================
    
    def _init_historical_distributions(self):
        """Initialise les distributions historiques des buts"""
        return {
            'goals_per_match': {
                '0': 0.08, '1': 0.13, '2': 0.18, '3': 0.22, '4': 0.16, '5': 0.10, '6+': 0.13
            },
            'btts_probability': 0.52,
            'over_15': 0.78,
            'over_25': 0.55,
            'over_35': 0.32,
            'halftime_goals': {
                '0': 0.35, '1': 0.30, '2': 0.20, '3+': 0.15
            },
            'secondhalf_goals': {
                '0': 0.25, '1': 0.28, '2': 0.22, '3+': 0.25
            },
            'both_halves_score': 0.28,
            'lead_at_ht_and_win': 0.72,
            'draw_at_ht_and_win': 0.18,
            'lose_at_ht_and_win': 0.10
        }
    
    def _init_league_averages(self):
        """Initialise les moyennes de buts par ligue"""
        return {
            'PL': {'home': 1.53, 'away': 1.18, 'total': 2.71},
            'PD': {'home': 1.48, 'away': 1.12, 'total': 2.60},
            'SA': {'home': 1.51, 'away': 1.21, 'total': 2.72},
            'BL1': {'home': 1.57, 'away': 1.29, 'total': 2.86},
            'FL1': {'home': 1.55, 'away': 1.19, 'total': 2.74},
            'CL': {'home': 1.62, 'away': 1.31, 'total': 2.93},
            'EL': {'home': 1.58, 'away': 1.24, 'total': 2.82},
            'EC': {'home': 1.49, 'away': 1.15, 'total': 2.64},
        }
    
    # ============================================
    # GESTION DES LIMITES API
    # ============================================
    
    def _rate_limited_request(self, url, params=None, max_retries=3):
        """Effectue une requête avec gestion des limites de taux"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if self.request_count >= 8 and time_since_last < 60:
            wait_time = 60 - time_since_last + 1
            print(f"   ⏳ Limite API approchée. Attente de {wait_time:.1f}s...")
            time.sleep(wait_time)
            self.request_count = 0
            self.last_request_time = time.time()
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                self.request_count += 1
                self.last_request_time = time.time()
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    wait_time = (attempt + 1) * 3
                    print(f"   ⏳ Limite API atteinte (429). Attente de {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return response
            except Exception as e:
                print(f"   ⚠️ Erreur requête: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return None
    
    # ============================================
    # NOUVELLES MÉTHODES: MÉTÉO
    # ============================================
    
    def get_weather(self, city, date):
        """
        Récupère les prévisions météo pour une ville
        Utilise Open-Meteo (gratuit, sans clé)
        """
        cache_key = f"{city}_{date}"
        
        if cache_key in self.weather_cache:
            return self.weather_cache[cache_key]
        
        try:
            # Coordonnées approximatives des grandes villes
            city_coords = {
                'London': (51.5074, -0.1278),
                'Manchester': (53.4808, -2.2426),
                'Liverpool': (53.4084, -2.9916),
                'Birmingham': (52.4862, -1.8904),
                'Madrid': (40.4168, -3.7038),
                'Barcelona': (41.3851, 2.1734),
                'Seville': (37.3891, -5.9845),
                'Valencia': (39.4699, -0.3763),
                'Rome': (41.9028, 12.4964),
                'Milan': (45.4642, 9.1900),
                'Turin': (45.0703, 7.6869),
                'Naples': (40.8518, 14.2681),
                'Berlin': (52.5200, 13.4050),
                'Munich': (48.1351, 11.5820),
                'Hamburg': (53.5511, 9.9937),
                'Paris': (48.8566, 2.3522),
                'Lyon': (45.7640, 4.8357),
                'Marseille': (43.2965, 5.3698),
            }
            
            # Chercher la ville dans le dictionnaire
            lat, lon = city_coords.get(city, (48.8566, 2.3522))  # Paris par défaut
            
            # Appel à l'API Open-Meteo
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': lat,
                'longitude': lon,
                'hourly': 'temperature_2m,precipitation,weathercode',
                'start_date': date,
                'end_date': date,
                'timezone': 'auto'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Prendre les données à 15h (heure typique des matchs)
                hour_index = 15
                if 'hourly' in data and len(data['hourly']['time']) > hour_index:
                    weather_result = {
                        'temperature': data['hourly']['temperature_2m'][hour_index],
                        'precipitation': data['hourly']['precipitation'][hour_index],
                        'weathercode': data['hourly']['weathercode'][hour_index],
                        'condition': self._decode_weather_code(data['hourly']['weathercode'][hour_index])
                    }
                    
                    self.weather_cache[cache_key] = weather_result
                    return weather_result
            
            return None
            
        except Exception as e:
            print(f"Erreur récupération météo: {e}")
            return None
    
    def _decode_weather_code(self, code):
        """Décode le code météo Open-Meteo en condition lisible"""
        if code == 0:
            return "☀️ Dégagé"
        elif code in [1, 2, 3]:
            return "⛅ Partiellement nuageux"
        elif code in [45, 48]:
            return "🌫️ Brouillard"
        elif code in [51, 53, 55]:
            return "🌧️ Bruine"
        elif code in [61, 63, 65]:
            return "🌧️ Pluie"
        elif code in [71, 73, 75]:
            return "❄️ Neige"
        elif code in [80, 81, 82]:
            return "🌧️ Averses"
        elif code in [95, 96, 99]:
            return "⛈️ Orage"
        else:
            return "🌡️ Inconnu"
    
    def _apply_weather_adjustments(self, markets, weather, match):
        """
        Ajuste les probabilités en fonction de la météo
        """
        if not weather:
            return markets
        
        # Pluie -> moins de buts
        if 'Pluie' in weather['condition'] or 'Averses' in weather['condition']:
            markets['total_goals']['over_25'] *= 0.9
            markets['total_goals']['under_25'] *= 1.1
            markets['btts']['oui'] *= 0.95
            markets['btts']['non'] *= 1.05
            
            # La pluie désavantage légèrement l'équipe visiteuse
            markets['1N2']['away'] *= 0.95
            markets['1N2']['home'] *= 1.02
        
        # Neige -> très peu de buts
        elif 'Neige' in weather['condition']:
            markets['total_goals']['over_25'] *= 0.7
            markets['total_goals']['under_25'] *= 1.3
            markets['btts']['oui'] *= 0.8
            markets['btts']['non'] *= 1.2
        
        # Vent fort (non disponible dans cette API simple)
        
        # Température froide (<5°C) -> moins de buts
        if weather['temperature'] < 5:
            markets['total_goals']['over_25'] *= 0.95
            markets['total_goals']['under_25'] *= 1.05
        
        # Température chaude (>25°C) -> légèrement plus de buts (fatigue)
        elif weather['temperature'] > 25:
            markets['total_goals']['over_25'] *= 1.05
            markets['total_goals']['under_25'] *= 0.95
        
        return markets
    
    # ============================================
    # NOUVELLES MÉTHODES: CONTEXTE DE COMPÉTITION
    # ============================================
    
    def get_league_standings(self, competition_code):
        """
        Récupère le classement d'une compétition
        Utilise l'API football-data.org
        """
        cache_key = f"standings_{competition_code}"
        
        if cache_key in self.standings_cache:
            return self.standings_cache[cache_key]
        
        try:
            url = f"{self.base_url}competitions/{competition_code}/standings"
            response = self._rate_limited_request(url)
            
            if response and response.status_code == 200:
                data = response.json()
                
                standings = {}
                for standing in data.get('standings', []):
                    if standing.get('type') == 'TOTAL':
                        for team in standing.get('table', []):
                            team_name = team['team']['name']
                            standings[team_name] = {
                                'position': team['position'],
                                'points': team['points'],
                                'played': team['playedGames'],
                                'won': team['won'],
                                'drawn': team['draw'],
                                'lost': team['lost'],
                                'goals_for': team['goalsFor'],
                                'goals_against': team['goalsAgainst']
                            }
                
                self.standings_cache[cache_key] = standings
                return standings
            
            return None
            
        except Exception as e:
            print(f"Erreur récupération classement: {e}")
            return None
    
    def analyze_motivation(self, match, standings):
        """
        Analyse la motivation des équipes en fonction du contexte
        """
        home_team = match['home_team']
        away_team = match['away_team']
        
        motivation = {
            'home': 1.0,
            'away': 1.0,
            'home_reason': [],
            'away_reason': []
        }
        
        if not standings:
            return motivation
        
        home_info = standings.get(home_team)
        away_info = standings.get(away_team)
        
        if home_info:
            pos = home_info['position']
            # Top 3 -> lutte pour le titre
            if pos <= 3:
                motivation['home'] *= 1.1
                motivation['home_reason'].append("Lutte pour le titre")
            # Top 5 -> lutte pour l'Europe
            elif pos <= 5:
                motivation['home'] *= 1.05
                motivation['home_reason'].append("Lutte pour l'Europe")
            # Bottom 3 -> lutte pour le maintien
            elif pos >= 18:
                motivation['home'] *= 1.08
                motivation['home_reason'].append("Lutte pour le maintien")
            # Milieu de tableau -> rien à jouer
            else:
                motivation['home'] *= 0.95
                motivation['home_reason'].append("Peu d'enjeu")
        
        if away_info:
            pos = away_info['position']
            if pos <= 3:
                motivation['away'] *= 1.1
                motivation['away_reason'].append("Lutte pour le titre")
            elif pos <= 5:
                motivation['away'] *= 1.05
                motivation['away_reason'].append("Lutte pour l'Europe")
            elif pos >= 18:
                motivation['away'] *= 1.08
                motivation['away_reason'].append("Lutte pour le maintien")
            else:
                motivation['away'] *= 0.95
                motivation['away_reason'].append("Peu d'enjeu")
        
        return motivation
    
    # ============================================
    # NOUVELLES MÉTHODES: FATIGUE DES JOUEURS
    # ============================================
    
    def get_team_recent_schedule(self, team_id, days=10):
        """
        Récupère le calendrier récent d'une équipe pour évaluer la fatigue
        """
        cache_key = f"schedule_{team_id}"
        
        if cache_key in self.team_schedule_cache:
            return self.team_schedule_cache[cache_key]
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_url}teams/{team_id}/matches"
            params = {
                'dateFrom': start_date.strftime('%Y-%m-%d'),
                'dateTo': end_date.strftime('%Y-%m-%d'),
                'status': 'FINISHED'
            }
            
            response = self._rate_limited_request(url, params)
            
            if response and response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                
                # Compter les matchs récents
                match_count = len(matches)
                
                # Vérifier s'il y a eu un match dans les 3 derniers jours
                recent_match = False
                three_days_ago = datetime.now() - timedelta(days=3)
                
                for match in matches:
                    match_date = datetime.strptime(match['utcDate'][:10], '%Y-%m-%d')
                    if match_date >= three_days_ago:
                        recent_match = True
                        break
                
                result = {
                    'matches_last_10_days': match_count,
                    'played_recently': recent_match,
                    'fatigue_factor': min(1.0, 1.0 - (match_count * 0.03))  # -3% par match
                }
                
                self.team_schedule_cache[cache_key] = result
                return result
            
            return None
            
        except Exception as e:
            print(f"Erreur récupération calendrier: {e}")
            return None
    
    def _apply_fatigue_adjustments(self, markets, home_schedule, away_schedule):
        """
        Ajuste les probabilités en fonction de la fatigue
        """
        if home_schedule:
            if home_schedule['played_recently']:
                markets['1N2']['home'] *= 0.95
                markets['total_goals']['over_25'] *= 0.97
            
            markets['1N2']['home'] *= home_schedule['fatigue_factor']
        
        if away_schedule:
            if away_schedule['played_recently']:
                markets['1N2']['away'] *= 0.95
                markets['total_goals']['over_25'] *= 0.97
            
            markets['1N2']['away'] *= away_schedule['fatigue_factor']
        
        return markets
    
    # ============================================
    # NOUVELLES MÉTHODES: ANALYSE GLOBALE DU CONTEXTE
    # ============================================
    
    def analyze_context(self, match):
        """
        Analyse globale du contexte du match
        """
        print(f"   📊 Analyse du contexte...")
        
        context = {
            'motivation': None,
            'weather': None,
            'fatigue': {},
            'standings': None
        }
        
        # Récupérer le classement
        standings = self.get_league_standings(match['competition'])
        if standings:
            context['standings'] = standings
            context['motivation'] = self.analyze_motivation(match, standings)
        
        # Récupérer la météo
        city = match['home_team'].split()[-1]  # Dernier mot du nom de l'équipe
        context['weather'] = self.get_weather(city, match['date'])
        
        # Analyser la fatigue
        home_schedule = self.get_team_recent_schedule(match['home_team_id'])
        away_schedule = self.get_team_recent_schedule(match['away_team_id'])
        
        context['fatigue'] = {
            'home': home_schedule,
            'away': away_schedule
        }
        
        return context
    
    def _apply_context_adjustments(self, markets, context):
        """
        Applique tous les ajustements contextuels
        """
        if not context:
            return markets
        
        # Ajustements météo
        if context.get('weather'):
            markets = self._apply_weather_adjustments(markets, context['weather'], None)
        
        # Ajustements motivation
        if context.get('motivation'):
            mot = context['motivation']
            markets['1N2']['home'] *= mot['home']
            markets['1N2']['away'] *= mot['away']
        
        # Ajustements fatigue
        if context.get('fatigue'):
            markets = self._apply_fatigue_adjustments(
                markets, 
                context['fatigue'].get('home'),
                context['fatigue'].get('away')
            )
        
        return markets
    
    # ============================================
    # MÉTHODES EXISTANTES: LINEUPS ET HEAD2HEAD
    # ============================================
    
    def get_match_lineups(self, match_id):
        """
        Récupère les compositions d'équipe pour un match
        """
        cache_key = f"lineups_{match_id}"
        
        if cache_key in self.lineups_cache:
            return self.lineups_cache[cache_key]
        
        try:
            url = f"{self.base_url}matches/{match_id}"
            response = self._rate_limited_request(url)
            
            if response and response.status_code == 200:
                data = response.json()
                
                # Extraire les compositions
                home_lineup = data.get('homeTeam', {}).get('lineup', [])
                away_lineup = data.get('awayTeam', {}).get('lineup', [])
                
                # Compter les titulaires par position
                home_starters = []
                away_starters = []
                
                # Analyser les titulaires à domicile
                for player in home_lineup:
                    if player.get('position') in ['Attacker', 'Midfielder']:
                        home_starters.append(player.get('name'))
                
                # Analyser les titulaires à l'extérieur
                for player in away_lineup:
                    if player.get('position') in ['Attacker', 'Midfielder']:
                        away_starters.append(player.get('name'))
                
                # Vérifier si les meilleurs buteurs sont titulaires
                home_scorers = self.scorers_cache.get('by_team', {}).get(data.get('homeTeam', {}).get('name'), [])
                away_scorers = self.scorers_cache.get('by_team', {}).get(data.get('awayTeam', {}).get('name'), [])
                
                home_top_scorer_playing = False
                away_top_scorer_playing = False
                
                if home_scorers and len(home_scorers) > 0:
                    top_scorer_name = home_scorers[0]['name']
                    home_top_scorer_playing = any(top_scorer_name in starter for starter in home_starters)
                
                if away_scorers and len(away_scorers) > 0:
                    top_scorer_name = away_scorers[0]['name']
                    away_top_scorer_playing = any(top_scorer_name in starter for starter in away_starters)
                
                result = {
                    'home_starters_count': len(home_starters),
                    'away_starters_count': len(away_starters),
                    'home_top_scorer_playing': home_top_scorer_playing,
                    'away_top_scorer_playing': away_top_scorer_playing,
                    'home_formation': data.get('homeTeam', {}).get('formation'),
                    'away_formation': data.get('awayTeam', {}).get('formation')
                }
                
                self.lineups_cache[cache_key] = result
                return result
                
            return None
            
        except Exception as e:
            print(f"Erreur récupération lineups: {e}")
            return None
    
    def get_head2head(self, match_id, limit=10):
        """
        Récupère l'historique des confrontations entre les deux équipes
        """
        cache_key = f"h2h_{match_id}"
        
        if cache_key in self.head2head_cache:
            return self.head2head_cache[cache_key]
        
        try:
            url = f"{self.base_url}matches/{match_id}/head2head"
            params = {'limit': limit}
            
            response = self._rate_limited_request(url, params)
            
            if response and response.status_code == 200:
                data = response.json()
                
                h2h_data = data.get('head2head', {})
                matches_list = data.get('matches', [])
                
                # Analyser les confrontations récentes
                recent_form = []
                for match in matches_list[:5]:
                    home_team = match.get('homeTeam', {}).get('name')
                    away_team = match.get('awayTeam', {}).get('name')
                    home_score = match.get('score', {}).get('fullTime', {}).get('home', 0)
                    away_score = match.get('score', {}).get('fullTime', {}).get('away', 0)
                    
                    if home_score > away_score:
                        winner = home_team
                    elif home_score < away_score:
                        winner = away_team
                    else:
                        winner = 'draw'
                    
                    recent_form.append({
                        'match': f"{home_team} vs {away_team}",
                        'score': f"{home_score}-{away_score}",
                        'winner': winner
                    })
                
                # Calculer les statistiques
                home_wins = h2h_data.get('homeTeam', {}).get('wins', 0)
                away_wins = h2h_data.get('awayTeam', {}).get('wins', 0)
                draws = h2h_data.get('draws', 0)
                total = home_wins + away_wins + draws
                
                # Calculer l'avantage psychologique
                if total > 0:
                    home_advantage = home_wins / total
                    away_advantage = away_wins / total
                else:
                    home_advantage = 0.33
                    away_advantage = 0.33
                
                result = {
                    'total_matches': total,
                    'home_wins': home_wins,
                    'away_wins': away_wins,
                    'draws': draws,
                    'home_advantage': round(home_advantage * 100, 1),
                    'away_advantage': round(away_advantage * 100, 1),
                    'recent_matches': recent_form[:5],
                    'home_goals_total': h2h_data.get('totalGoals', {}).get('homeTeam', 0),
                    'away_goals_total': h2h_data.get('totalGoals', {}).get('awayTeam', 0)
                }
                
                self.head2head_cache[cache_key] = result
                return result
                
            return None
            
        except Exception as e:
            print(f"Erreur récupération head2head: {e}")
            return None
    
    def _apply_lineups_adjustments(self, markets, lineups, match):
        """
        Ajuste les probabilités en fonction des compositions
        """
        if not lineups:
            return markets
        
        # Si le meilleur buteur à domicile n'est pas titulaire
        if not lineups.get('home_top_scorer_playing', True):
            markets['1N2']['home'] *= 0.85  # -15%
            markets['btts']['oui'] *= 0.85  # -15%
            markets['total_goals']['over_25'] *= 0.85
        
        # Si le meilleur buteur à l'extérieur n'est pas titulaire
        if not lineups.get('away_top_scorer_playing', True):
            markets['1N2']['away'] *= 0.85  # -15%
            markets['btts']['oui'] *= 0.85  # -15%
            markets['total_goals']['over_25'] *= 0.85
        
        # Si peu d'attaquants titulaires
        if lineups.get('home_starters_count', 10) < 3:
            markets['1N2']['home'] *= 0.9
            markets['btts']['oui'] *= 0.9
        
        if lineups.get('away_starters_count', 10) < 3:
            markets['1N2']['away'] *= 0.9
            markets['btts']['oui'] *= 0.9
        
        return markets
    
    def _apply_head2head_adjustments(self, markets, h2h, match):
        """
        Ajuste les probabilités en fonction de l'historique
        """
        if not h2h or h2h['total_matches'] < 3:
            return markets
        
        # Si une équipe domine historiquement
        if h2h['home_advantage'] > 70:  # Plus de 70% de victoires à domicile dans l'historique
            markets['1N2']['home'] *= 1.1  # +10%
            markets['1N2']['away'] *= 0.9
        
        if h2h['away_advantage'] > 70:  # Plus de 70% de victoires à l'extérieur dans l'historique
            markets['1N2']['away'] *= 1.1  # +10%
            markets['1N2']['home'] *= 0.9
        
        # Si beaucoup de matchs nuls dans l'historique
        if h2h['draws'] > h2h['total_matches'] * 0.4:  # Plus de 40% de nuls
            markets['1N2']['draw'] *= 1.15  # +15%
        
        # Si beaucoup de buts dans l'historique
        if h2h['total_matches'] > 0:
            avg_goals = (h2h['home_goals_total'] + h2h['away_goals_total']) / h2h['total_matches']
            if avg_goals > 3.0:
                markets['total_goals']['over_25'] *= 1.1
            elif avg_goals < 2.0:
                markets['total_goals']['under_25'] *= 1.1
        
        return markets
    
    # ============================================
    # MÉTHODES API EXISTANTES
    # ============================================
    
    def test_api_connection(self):
        """Teste la connexion à l'API"""
        try:
            response = self._rate_limited_request(f"{self.base_url}competitions")
            return response is not None and response.status_code == 200
        except Exception as e:
            print(f"Erreur connexion API: {e}")
            return False
    
    def display_available_competitions(self):
        """Affiche la liste des compétitions disponibles"""
        print("\n" + "="*60)
        print("🏆 COMPÉTITIONS DISPONIBLES")
        print("="*60)
        
        competitions_by_country = {}
        for code, info in self.competitions.items():
            country = info['country']
            if country not in competitions_by_country:
                competitions_by_country[country] = []
            competitions_by_country[country].append((code, info['name']))
        
        for country, comps in competitions_by_country.items():
            print(f"\n📌 {country}:")
            for code, name in comps:
                print(f"   {code}: {name}")
        
        print("\n" + "="*60)
    
    def fetch_historical_results(self, competition_code, months=2):
        """Récupère les résultats historiques pour l'entraînement"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30*months)
        
        url = f"{self.base_url}competitions/{competition_code}/matches"
        params = {
            'dateFrom': start_date.strftime('%Y-%m-%d'),
            'dateTo': end_date.strftime('%Y-%m-%d'),
            'status': 'FINISHED'
        }
        
        response = self._rate_limited_request(url, params)
        if response and response.status_code == 200:
            data = response.json()
            return data.get('matches', [])
        return []
    
    def fetch_upcoming_matches(self, competition_code, days_ahead=7):
        """Récupère les matchs à venir"""
        today = datetime.now().strftime('%Y-%m-%d')
        future_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        print(f"🔍 Recherche matchs du {today} au {future_date} pour {competition_code}...")
        
        try:
            url = f"{self.base_url}competitions/{competition_code}/matches"
            params = {'dateFrom': today, 'dateTo': future_date}
            
            response = self._rate_limited_request(url, params)
            
            if response and response.status_code == 200:
                data = response.json()
                all_matches = data.get('matches', [])
                print(f"   {len(all_matches)} matchs trouvés")
                
                upcoming_matches = []
                for match in all_matches:
                    status = match.get('status', '')
                    if status in ['SCHEDULED', 'TIMED']:
                        if match['homeTeam']['id'] and match['awayTeam']['id']:
                            upcoming_matches.append(match)
                
                print(f"   {len(upcoming_matches)} matchs programmés avec IDs valides")
                
                enriched_matches = []
                for i, match in enumerate(upcoming_matches):
                    if i > 0 and i % 3 == 0:
                        print(f"   ⏳ Pause pour éviter limite API...")
                        time.sleep(2)
                    
                    enriched_match = self._enrich_match_data(match)
                    if enriched_match:
                        enriched_matches.append(enriched_match)
                
                return enriched_matches
            else:
                return []
                
        except Exception as e:
            print(f"   Exception: {e}")
            return []
    
    def fetch_todays_matches(self, competition_code):
        """Récupère les matchs du jour"""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.fetch_upcoming_matches(competition_code, days_ahead=0)
    
    def _enrich_match_data(self, match):
        """Enrichit les données du match"""
        try:
            home_team_id = match['homeTeam']['id']
            away_team_id = match['awayTeam']['id']
            competition_code = match['competition']['code']
            
            if not home_team_id or not away_team_id:
                return None
            
            home_stats = self._get_team_stats(home_team_id, competition_code)
            away_stats = self._get_team_stats(away_team_id, competition_code)
            
            enriched = {
                'id': match['id'],
                'competition': competition_code,
                'competition_name': match['competition']['name'],
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'date': match['utcDate'][:10],
                'time': match['utcDate'][11:16],
                'stage': match.get('stage', 'REGULAR_SEASON'),
                'status': match.get('status', 'SCHEDULED'),
                'home_form': home_stats.get('form', [3, 3, 3, 3, 3]) if home_stats else [3, 3, 3, 3, 3],
                'away_form': away_stats.get('form', [3, 3, 3, 3, 3]) if away_stats else [3, 3, 3, 3, 3],
                'home_goals_scored_avg': home_stats.get('goals_scored_avg', 1.5) if home_stats else 1.5,
                'home_goals_conceded_avg': home_stats.get('goals_conceded_avg', 1.2) if home_stats else 1.2,
                'away_goals_scored_avg': away_stats.get('goals_scored_avg_away', 1.3) if away_stats else 1.3,
                'away_goals_conceded_avg': away_stats.get('goals_conceded_avg_away', 1.4) if away_stats else 1.4,
                'home_position': home_stats.get('position', 10) if home_stats else 10,
                'away_position': away_stats.get('position', 10) if away_stats else 10,
            }
            
            return enriched
            
        except Exception as e:
            return None
    
    def _get_team_stats(self, team_id, competition_code):
        """Récupère les statistiques d'une équipe"""
        if not team_id or team_id == 'None':
            return None
            
        cache_key = f"{team_id}_{competition_code}"
        
        if cache_key in self.teams_cache:
            return self.teams_cache[cache_key]
        
        try:
            url = f"{self.base_url}teams/{team_id}/matches"
            params = {
                'competitions': competition_code,
                'status': 'FINISHED',
                'limit': 10
            }
            
            response = self._rate_limited_request(url, params)
            
            if response and response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                
                if matches:
                    stats = self._calculate_team_stats(matches, team_id)
                    self.teams_cache[cache_key] = stats
                    
                    for match in matches:
                        self._store_historical_result(match, team_id)
                    
                    return stats
            return None
            
        except Exception as e:
            return None
    
    def _calculate_team_stats(self, matches, team_id):
        """Calcule les statistiques d'une équipe"""
        form = []
        goals_scored = []
        goals_conceded = []
        goals_scored_away = []
        goals_conceded_away = []
        
        for match in matches[:5]:
            try:
                is_home = match['homeTeam']['id'] == team_id
                
                if is_home:
                    scored = match['score']['fullTime']['home']
                    conceded = match['score']['fullTime']['away']
                else:
                    scored = match['score']['fullTime']['away']
                    conceded = match['score']['fullTime']['home']
                    goals_scored_away.append(scored)
                    goals_conceded_away.append(conceded)
                
                goals_scored.append(scored)
                goals_conceded.append(conceded)
                
                if scored > conceded:
                    form.append(3)
                elif scored == conceded:
                    form.append(1)
                else:
                    form.append(0)
            except:
                continue
        
        return {
            'form': form[::-1] if form else [3, 3, 3, 3, 3],
            'goals_scored_avg': np.mean(goals_scored) if goals_scored else 1.5,
            'goals_conceded_avg': np.mean(goals_conceded) if goals_conceded else 1.2,
            'goals_scored_avg_away': np.mean(goals_scored_away) if goals_scored_away else 1.3,
            'goals_conceded_avg_away': np.mean(goals_conceded_away) if goals_conceded_away else 1.4,
            'position': 10
        }
    
    def _store_historical_result(self, match, team_id):
        """Stocke un résultat historique"""
        try:
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            home_score = match['score']['fullTime']['home']
            away_score = match['score']['fullTime']['away']
            
            key = f"{home_team}_{away_team}_{match['utcDate'][:10]}"
            
            if home_score > away_score:
                result = 0
            elif home_score == away_score:
                result = 1
            else:
                result = 2
            
            self.historical_results[key] = {
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'result': result,
                'date': match['utcDate'][:10]
            }
        except:
            pass
    
    # ============================================
    # MÉTHODES DE CALCUL
    # ============================================
    
    def _calculate_form_score(self, form_list):
        """Calcule un score de forme pondéré"""
        if not form_list:
            return 0
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        return sum(f * w for f, w in zip(form_list[:5], weights[:len(form_list)]))
    
    def _calculate_attack_defense_ratio(self, goals_scored, goals_conceded):
        """Calcule le ratio attaque/défense"""
        if goals_conceded == 0:
            return goals_scored * 2
        return goals_scored / goals_conceded
    
    def _create_match_features(self, match):
        """Crée les caractéristiques pour un match"""
        features = {}
        
        features['home_form_score'] = self._calculate_form_score(match.get('home_form', []))
        features['away_form_score'] = self._calculate_form_score(match.get('away_form', []))
        features['form_difference'] = features['home_form_score'] - features['away_form_score']
        
        features['home_attack_ratio'] = self._calculate_attack_defense_ratio(
            match.get('home_goals_scored_avg', 1), 
            match.get('home_goals_conceded_avg', 1)
        )
        features['away_attack_ratio'] = self._calculate_attack_defense_ratio(
            match.get('away_goals_scored_avg', 1), 
            match.get('away_goals_conceded_avg', 1)
        )
        features['attack_ratio_diff'] = features['home_attack_ratio'] - features['away_attack_ratio']
        
        features['position_advantage'] = match.get('away_position', 10) - match.get('home_position', 10)
        
        return features
    
    # ============================================
    # ENTRAÎNEMENT
    # ============================================
    
    def train_models(self):
        """Entraîne les modèles ML avec des données simulées (fallback)"""
        print("\n🤖 Entraînement des modèles (mode fallback)...")
        
        np.random.seed(42)
        n_samples = 500
        n_features = 10
        
        X_train = np.random.rand(n_samples, n_features)
        y_train = np.random.randint(0, 3, n_samples)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
            print(f"  ✅ {name}: Précision = {accuracy:.2f}")
        
        self.is_trained = True
        print("✅ Modèles entraînés avec succès!\n")
    
    def train_with_historical_data(self, competition_codes=None):
        """Entraîne les modèles avec des données historiques réelles (optimisé)"""
        print("\n📚 ENTRAÎNEMENT AVEC DONNÉES HISTORIQUES RÉELLES")
        print("="*60)
        
        if not competition_codes:
            competition_codes = list(self.competitions.keys())
        
        all_features = []
        all_results = []
        total_matches = 0
        
        for comp in competition_codes:
            print(f"\n📊 Récupération des données pour {self.competitions[comp]['name']}...")
            matches = self.fetch_historical_results(comp, months=2)
            
            if matches:
                print(f"   {len(matches)} matchs trouvés")
                total_matches += len(matches)
                
                for i, match in enumerate(matches):
                    try:
                        # Ne traiter que 50% des matchs pour accélérer
                        if i % 2 == 0:
                            enriched = self._enrich_match_data(match)
                            if enriched:
                                features = self._create_match_features(enriched)
                                
                                home_score = match['score']['fullTime']['home']
                                away_score = match['score']['fullTime']['away']
                                
                                if home_score > away_score:
                                    result = 0
                                elif home_score == away_score:
                                    result = 1
                                else:
                                    result = 2
                                
                                all_features.append(list(features.values()))
                                all_results.append(result)
                                
                    except Exception as e:
                        continue
                    
                    # Petite pause tous les 10 matchs
                    if i > 0 and i % 10 == 0:
                        time.sleep(1)
        
        print(f"\n📊 Total: {total_matches} matchs analysés")
        
        if len(all_features) > 20:
            print(f"✅ {len(all_features)} échantillons d'entraînement collectés")
            
            X = np.array(all_features)
            y = np.array(all_results)
            
            X_scaled = self.scaler.fit_transform(X)
            
            print("\n🤖 Entraînement des modèles...")
            for name, model in self.models.items():
                model.fit(X_scaled, y)
                accuracy = accuracy_score(y, model.predict(X_scaled))
                print(f"  ✅ {name}: Précision = {accuracy:.2f}")
            
            self.is_trained = True
            self._save_trained_models()
            print("\n✅ Modèles entraînés avec succès!")
        else:
            print(f"\n⚠️ Pas assez de données ({len(all_features)}), utilisation du mode fallback")
            self.train_models()
    
    def train_poisson_model(self, matches):
        """Entraîne le modèle de Poisson"""
        print("\n📊 Entraînement du modèle de Poisson...")
        
        teams_strength = {}
        
        for match in matches:
            home_team = match['home_team']
            away_team = match['away_team']
            competition = match['competition']
            
            if home_team not in teams_strength:
                teams_strength[home_team] = {
                    'attack_home': [], 'defense_home': [],
                    'attack_away': [], 'defense_away': []
                }
            if away_team not in teams_strength:
                teams_strength[away_team] = {
                    'attack_home': [], 'defense_home': [],
                    'attack_away': [], 'defense_away': []
                }
            
            if 'home_goals_scored_avg' in match:
                league_avg = self.league_averages.get(competition, {'home': 1.5, 'away': 1.2})
                
                teams_strength[home_team]['attack_home'].append(match['home_goals_scored_avg'] / league_avg['home'])
                teams_strength[home_team]['defense_home'].append(match['home_goals_conceded_avg'] / league_avg['away'])
                teams_strength[away_team]['attack_away'].append(match['away_goals_scored_avg'] / league_avg['away'])
                teams_strength[away_team]['defense_away'].append(match['away_goals_conceded_avg'] / league_avg['home'])
        
        self.team_attack_strength = {}
        self.team_defense_strength = {}
        
        for team, strengths in teams_strength.items():
            self.team_attack_strength[team] = {
                'home': np.mean(strengths['attack_home']) if strengths['attack_home'] else 1.0,
                'away': np.mean(strengths['attack_away']) if strengths['attack_away'] else 1.0
            }
            self.team_defense_strength[team] = {
                'home': np.mean(strengths['defense_home']) if strengths['defense_home'] else 1.0,
                'away': np.mean(strengths['defense_away']) if strengths['defense_away'] else 1.0
            }
        
        print(f"✅ Modèle Poisson entraîné avec {len(self.team_attack_strength)} équipes")
    
    # ============================================
    # PRÉDICTION DES SCORES EXACTS
    # ============================================
    
    def predict_poisson_scores(self, match):
        """Prédit les scores exacts avec Poisson"""
        home_team = match['home_team']
        away_team = match['away_team']
        competition = match['competition']
        
        league_avg = self.league_averages.get(competition, {'home': 1.5, 'away': 1.2})
        
        home_attack = self.team_attack_strength.get(home_team, {}).get('home', 1.0)
        home_defense = self.team_defense_strength.get(home_team, {}).get('home', 1.0)
        away_attack = self.team_attack_strength.get(away_team, {}).get('away', 1.0)
        away_defense = self.team_defense_strength.get(away_team, {}).get('away', 1.0)
        
        home_xG = league_avg['home'] * home_attack * away_defense * 1.1
        away_xG = league_avg['away'] * away_attack * home_defense * 0.95
        
        score_matrix = {}
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        for h in range(0, 8):
            for a in range(0, 8):
                prob = poisson.pmf(h, home_xG) * poisson.pmf(a, away_xG)
                score_matrix[f"{h}-{a}"] = prob
                
                if h > a:
                    home_win_prob += prob
                elif h == a:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        total = home_win_prob + draw_prob + away_win_prob
        if total > 0:
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total
        
        top_scores = sorted(score_matrix.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'home_xG': home_xG,
            'away_xG': away_xG,
            'expected_total': home_xG + away_xG,
            'score_matrix': score_matrix,
            'top_scores': top_scores,
            'probabilities': {
                '1': home_win_prob,
                'N': draw_prob,
                '2': away_win_prob
            }
        }
    
    # ============================================
    # ANALYSE DES MARCHÉS (VERSION ULTIME AVEC CONTEXTE)
    # ============================================
    
    def analyze_all_markets(self, match):
        """Analyse tous les marchés disponibles avec lineups, head2head et contexte"""
        print(f"\n🔍 Analyse ULTIME de {match['home_team']} vs {match['away_team']}...")
        
        # Prédiction de base Poisson
        poisson_results = self.predict_poisson_scores(match)
        
        # Récupérer les données supplémentaires
        print(f"   📋 Récupération des compositions...")
        lineups = self.get_match_lineups(match['id'])
        
        print(f"   📜 Récupération de l'historique...")
        h2h = self.get_head2head(match['id'])
        
        print(f"   🌍 Analyse du contexte...")
        context = self.analyze_context(match)
        
        # Initialisation des résultats
        markets = {}
        
        # 1. 1N2
        markets['1N2'] = {
            'home': round(poisson_results['probabilities']['1'] * 100, 1),
            'draw': round(poisson_results['probabilities']['N'] * 100, 1),
            'away': round(poisson_results['probabilities']['2'] * 100, 1),
            'recommendation': max(poisson_results['probabilities'], key=poisson_results['probabilities'].get),
            'confidence': round(max(poisson_results['probabilities'].values()) * 100, 1)
        }
        
        # 2. DOUBLE CHANCE
        markets['double_chance'] = {
            '1N': round((poisson_results['probabilities']['1'] + poisson_results['probabilities']['N']) * 100, 1),
            'N2': round((poisson_results['probabilities']['N'] + poisson_results['probabilities']['2']) * 100, 1),
            '12': round((poisson_results['probabilities']['1'] + poisson_results['probabilities']['2']) * 100, 1),
        }
        
        # 3. BTTS
        btts_prob = 0
        for score, prob in poisson_results['score_matrix'].items():
            h, a = map(int, score.split('-'))
            if h > 0 and a > 0:
                btts_prob += prob
        
        markets['btts'] = {
            'oui': round(btts_prob * 100, 1),
            'non': round((1 - btts_prob) * 100, 1),
        }
        
        # 4. TOTAL DES BUTS
        markets['total_goals'] = {
            'expected': round(poisson_results['expected_total'], 2),
            'under_05': round(self._calculate_over_under_prob(poisson_results['score_matrix'], 0.5, 'under') * 100, 1),
            'over_05': round(self._calculate_over_under_prob(poisson_results['score_matrix'], 0.5, 'over') * 100, 1),
            'under_15': round(self._calculate_over_under_prob(poisson_results['score_matrix'], 1.5, 'under') * 100, 1),
            'over_15': round(self._calculate_over_under_prob(poisson_results['score_matrix'], 1.5, 'over') * 100, 1),
            'under_25': round(self._calculate_over_under_prob(poisson_results['score_matrix'], 2.5, 'under') * 100, 1),
            'over_25': round(self._calculate_over_under_prob(poisson_results['score_matrix'], 2.5, 'over') * 100, 1),
            'under_35': round(self._calculate_over_under_prob(poisson_results['score_matrix'], 3.5, 'under') * 100, 1),
            'over_35': round(self._calculate_over_under_prob(poisson_results['score_matrix'], 3.5, 'over') * 100, 1),
        }
        
        # 5. SCORES EXACTS
        markets['exact_scores'] = [
            {'score': score, 'probability': round(prob * 100, 1)}
            for score, prob in poisson_results['top_scores'][:5]
        ]
        
        # 6. MI-TEMPS
        markets['halftime_goals'] = self._calculate_halftime_goals_probabilities(match, poisson_results)
        
        # 7. COMPARAISON MI-TEMPS
        markets['halftime_comparison'] = self._calculate_halftime_comparison(match)
        
        # 8. APPLIQUER LES AJUSTEMENTS (LINEUPS, HEAD2HEAD ET CONTEXTE)
        markets = self._apply_lineups_adjustments(markets, lineups, match)
        markets = self._apply_head2head_adjustments(markets, h2h, match)
        markets = self._apply_context_adjustments(markets, context)
        
        # 9. CONFIDENCE GLOBALE
        markets['global_confidence'] = self._calculate_global_confidence(markets)
        
        # 10. AJOUTER LES INFORMATIONS SUPPLÉMENTAIRES
        markets['lineups_info'] = lineups
        markets['head2head_info'] = h2h
        markets['context_info'] = {
            'weather': context.get('weather'),
            'motivation': context.get('motivation'),
            'fatigue': context.get('fatigue')
        }
        
        return markets
    
    def _calculate_over_under_prob(self, score_matrix, threshold, direction):
        """Calcule la probabilité over/under"""
        prob = 0
        for score, p in score_matrix.items():
            h, a = map(int, score.split('-'))
            total = h + a
            if direction == 'over' and total > threshold:
                prob += p
            elif direction == 'under' and total < threshold:
                prob += p
        return prob
    
    def _calculate_halftime_goals_probabilities(self, match, poisson_results):
        """Calcule les probabilités pour les buts en première période"""
        home_ht_xG = poisson_results['home_xG'] * 0.45
        away_ht_xG = poisson_results['away_xG'] * 0.45
        total_ht_xG = home_ht_xG + away_ht_xG
        
        ht_over_05 = 1 - poisson.pmf(0, total_ht_xG)
        ht_over_15 = 1 - (poisson.pmf(0, total_ht_xG) + poisson.pmf(1, total_ht_xG))
        
        return {
            'expected_goals': round(total_ht_xG, 2),
            'over_05': round(ht_over_05 * 100, 1),
            'over_15': round(ht_over_15 * 100, 1),
        }
    
    def _calculate_halftime_comparison(self, match):
        """Compare les buts en première et seconde période"""
        home_strength = match.get('home_goals_scored_avg', 1.5) / 1.5
        away_strength = match.get('away_goals_scored_avg', 1.3) / 1.3
        total_strength = (home_strength + away_strength) / 2
        
        first_half_more = 0.35 * total_strength
        second_half_more = 0.45 * total_strength
        equal_halves = 0.20 * total_strength
        
        total = first_half_more + second_half_more + equal_halves
        
        return {
            'first_half_more': round(first_half_more / total * 100, 1),
            'second_half_more': round(second_half_more / total * 100, 1),
            'equal_halves': round(equal_halves / total * 100, 1),
        }
    
    def _calculate_global_confidence(self, markets):
        """Calcule un score de confiance global"""
        scores = []
        scores.append(markets['1N2']['confidence'])
        scores.append(max(markets['btts']['oui'], markets['btts']['non']))
        scores.append(max(markets['total_goals']['over_25'], markets['total_goals']['under_25']))
        
        avg_confidence = sum(scores) / len(scores)
        
        return {
            'average': round(avg_confidence, 1),
            'level': 'TRÈS HAUTE' if avg_confidence > 75 else 'HAUTE' if avg_confidence > 60 else 'MOYENNE'
        }
    
    # ============================================
    # IDENTIFICATION DES PARIS SÛRS (AMÉLIORÉE)
    # ============================================
    
    def identify_safe_bets(self, markets):
        """Identifie les éléments les plus sûrs"""
        safe_bets = []
        
        # 1N2
        home_prob = markets['1N2']['home']
        draw_prob = markets['1N2']['draw']
        away_prob = markets['1N2']['away']
        
        if home_prob > 65:
            safe_bets.append({
                'type': '1N2',
                'prediction': 'VICTOIRE DOMICILE',
                'probability': home_prob,
                'confidence': 'TRÈS HAUTE' if home_prob > 75 else 'HAUTE',
                'emoji': '🏠',
                'reason': f"L'équipe à domicile a {home_prob}% de chances de gagner"
            })
        elif away_prob > 65:
            safe_bets.append({
                'type': '1N2',
                'prediction': 'VICTOIRE EXTERIEUR',
                'probability': away_prob,
                'confidence': 'TRÈS HAUTE' if away_prob > 75 else 'HAUTE',
                'emoji': '✈️',
                'reason': f"L'équipe à l'extérieur a {away_prob}% de chances de gagner"
            })
        elif draw_prob > 40:
            safe_bets.append({
                'type': '1N2',
                'prediction': 'MATCH NUL',
                'probability': draw_prob,
                'confidence': 'MOYENNE',
                'emoji': '🤝',
                'reason': f"Le match nul est probable à {draw_prob}%"
            })
        
        # DOUBLE CHANCE
        if markets['double_chance']['1N'] > 85:
            safe_bets.append({
                'type': 'DOUBLE CHANCE',
                'prediction': '1N (Domicile ou Nul)',
                'probability': markets['double_chance']['1N'],
                'confidence': 'TRÈS HAUTE',
                'emoji': '🛡️',
                'reason': "L'équipe à domicile est très solide"
            })
        elif markets['double_chance']['N2'] > 85:
            safe_bets.append({
                'type': 'DOUBLE CHANCE',
                'prediction': 'N2 (Extérieur ou Nul)',
                'probability': markets['double_chance']['N2'],
                'confidence': 'TRÈS HAUTE',
                'emoji': '🛡️',
                'reason': "L'équipe à l'extérieur est très solide"
            })
        elif markets['double_chance']['12'] > 85:
            safe_bets.append({
                'type': 'DOUBLE CHANCE',
                'prediction': '12 (Pas de nul)',
                'probability': markets['double_chance']['12'],
                'confidence': 'TRÈS HAUTE',
                'emoji': '⚔️',
                'reason': "Les deux équipes sont offensives"
            })
        
        # BTTS
        btts_oui = markets['btts']['oui']
        btts_non = markets['btts']['non']
        
        if btts_oui > 70:
            safe_bets.append({
                'type': 'BTTS',
                'prediction': 'LES DEUX ÉQUIPES MARQUENT',
                'probability': btts_oui,
                'confidence': 'TRÈS HAUTE' if btts_oui > 80 else 'HAUTE',
                'emoji': '⚽⚽',
                'reason': "Les deux équipes ont une forte capacité offensive"
            })
        elif btts_non > 70:
            safe_bets.append({
                'type': 'BTTS',
                'prediction': 'PAS DE BUT DES DEUX CÔTÉS',
                'probability': btts_non,
                'confidence': 'TRÈS HAUTE' if btts_non > 80 else 'HAUTE',
                'emoji': '🔒',
                'reason': "Au moins une équipe aura du mal à marquer"
            })
        
        # OVER/UNDER
        over_25 = markets['total_goals']['over_25']
        under_25 = markets['total_goals']['under_25']
        expected = markets['total_goals']['expected']
        
        if over_25 > 70:
            safe_bets.append({
                'type': 'OVER 2.5',
                'prediction': 'PLUS DE 2.5 BUTS',
                'probability': over_25,
                'confidence': 'TRÈS HAUTE' if over_25 > 80 else 'HAUTE',
                'emoji': '📈',
                'reason': f"Match offensif attendu avec {expected} buts prévus"
            })
        elif under_25 > 70:
            safe_bets.append({
                'type': 'UNDER 2.5',
                'prediction': 'MOINS DE 2.5 BUTS',
                'probability': under_25,
                'confidence': 'TRÈS HAUTE' if under_25 > 80 else 'HAUTE',
                'emoji': '📉',
                'reason': f"Match fermé attendu avec {expected} buts prévus"
            })
        
        # OVER 0.5
        over_05 = markets['total_goals']['over_05']
        if over_05 > 90:
            safe_bets.append({
                'type': 'OVER 0.5',
                'prediction': 'AU MOINS UN BUT',
                'probability': over_05,
                'confidence': 'EXTRÊME',
                'emoji': '🎯',
                'reason': "Il est extrêmement probable qu'il y ait au moins un but"
            })
        
        # MI-TEMPS
        ht_over_05 = markets['halftime_goals']['over_05']
        if ht_over_05 > 75:
            safe_bets.append({
                'type': 'MI-TEMPS',
                'prediction': 'BUT EN PREMIÈRE MI-TEMPS',
                'probability': ht_over_05,
                'confidence': 'HAUTE',
                'emoji': '⏱️',
                'reason': "Fortes chances d'avoir un but dès la première période"
            })
        
        # TRI
        safe_bets.sort(key=lambda x: x['probability'], reverse=True)
        
        return safe_bets
    
    # ============================================
    # SÉLECTION DES MEILLEURS MATCHS
    # ============================================
    
    def select_best_matches(self, matches, max_matches=5):
        """Sélectionne les meilleurs matchs"""
        analyzed = []
        
        for i, match in enumerate(matches):
            if i > 0 and i % 2 == 0:
                time.sleep(1)
            
            try:
                markets = self.analyze_all_markets(match)
                confidence = markets['global_confidence']['average']
                
                analyzed.append({
                    'match': match,
                    'markets': markets,
                    'confidence': confidence
                })
                
                print(f"   ✅ Match analysé: {match['home_team']} vs {match['away_team']} - Conf: {confidence:.1f}%")
                
            except Exception as e:
                print(f"   ⚠️ Erreur analyse: {e}")
                continue
        
        analyzed.sort(key=lambda x: x['confidence'], reverse=True)
        return analyzed[:min(max_matches, len(analyzed))]


# ============================================
# PROGRAMME PRINCIPAL
# ============================================

def main():
    """Point d'entrée principal"""
    if os.name == 'nt':
        import subprocess
        subprocess.run('chcp 65001', shell=True, capture_output=True)
    
    print("="*60)
    print("🏆 FOOTBALL MATCH SELECTOR ULTRA - DÉMARRAGE")
    print("="*60)
    
    selector = FootballMatchSelectorUltra(api_key="1b9fa9eead33409cb75f3d0a2df60324")
    
    if selector.test_api_connection():
        print("✅ Connexion API réussie!")
        
        if not selector.is_trained:
            print("\n📚 Entraînement des modèles...")
            selector.train_with_historical_data(['FL1', 'CL', 'EL', 'EC'])
        
        print("\n🔍 ANALYSE DES MATCHS À VENIR")
        print("="*60)
        
        # Analyser plusieurs compétitions
        competitions_to_analyze = ['FL1', 'PD', 'PL', 'SA', 'BL1', 'CL']
        
        for comp in competitions_to_analyze:
            print(f"\n📊 Compétition: {selector.competitions[comp]['name']}")
            matches = selector.fetch_upcoming_matches(comp, days_ahead=7)
            
            if matches:
                print(f"   {len(matches)} matchs trouvés")
                
                # Sélectionner les meilleurs matchs
                best_matches = selector.select_best_matches(matches, max_matches=2)
                
                for i, match_data in enumerate(best_matches):
                    match = match_data['match']
                    markets = match_data['markets']
                    
                    print(f"\n{'='*60}")
                    print(f"🏆 MATCH #{i+1}: {match['home_team']} vs {match['away_team']}")
                    print(f"📅 {match['date']} à {match['time']}")
                    print(f"🏟️ {match['competition_name']}")
                    print(f"📊 Confiance globale: {markets['global_confidence']['average']}% ({markets['global_confidence']['level']})")
                    
                    # Afficher les paris sûrs
                    safe_bets = selector.identify_safe_bets(markets)
                    if safe_bets:
                        print("\n🎯 PARIS RECOMMANDÉS:")
                        for bet in safe_bets[:3]:
                            print(f"   {bet['emoji']} {bet['prediction']}: {bet['probability']}% ({bet['confidence']})")
                            print(f"      ➡️ {bet['reason']}")
                    
                    # Afficher les scores exacts probables
                    if 'exact_scores' in markets:
                        print("\n📊 SCORES EXACTS PROBABLES:")
                        for score in markets['exact_scores'][:3]:
                            print(f"   {score['score']}: {score['probability']}%")
                    
                    # Afficher les infos contexte si disponibles
                    if markets.get('context_info'):
                        print("\n🌍 CONTEXTE:")
                        if markets['context_info'].get('weather'):
                            weather = markets['context_info']['weather']
                            if weather:
                                print(f"   Météo: {weather.get('condition', 'N/A')}, {weather.get('temperature', '?')}°C")
                        
                        if markets.get('lineups_info'):
                            lineups = markets['lineups_info']
                            if lineups:
                                print(f"   Meilleur buteur domicile présent: {'✅' if lineups.get('home_top_scorer_playing') else '❌'}")
                                print(f"   Meilleur buteur extérieur présent: {'✅' if lineups.get('away_top_scorer_playing') else '❌'}")
                    
                    print(f"\n   {'='*50}")
                    
                    # Pause entre les matchs
                    time.sleep(2)
            else:
                print(f"   ⚠️ Aucun match trouvé pour cette compétition")
            
            # Pause entre les compétitions
            time.sleep(3)
        
        print("\n" + "="*60)
        print("✅ ANALYSE TERMINÉE - Rafraîchissez la page dans 1 heure")
        print("="*60)
        
    else:
        print("❌ Échec connexion API")
        print("   Vérifiez votre clé API et votre connexion internet")

if __name__ == "__main__":
    main()