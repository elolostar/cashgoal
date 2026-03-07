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
    Version OPTIMISÉE pour Koyeb
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
            'random_forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),  # Réduit
            'gradient_boosting': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42),  # Réduit
            'logistic_regression': LogisticRegression(max_iter=500, random_state=42)  # Réduit
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        
        # Compétitions disponibles (garder seulement l'essentiel)
        self.competitions = {
            'FL1': {'name': 'Ligue 1 (France)', 'country': 'France', 'code': 'FL1'},
            'CL': {'name': 'UEFA Champions League', 'country': 'Europe', 'code': 'CL'},
            'PL': {'name': 'Premier League (Angleterre)', 'country': 'Angleterre', 'code': 'PL'},
            'PD': {'name': 'La Liga (Espagne)', 'country': 'Espagne', 'code': 'PD'},
        }
        
        # Cache
        self.teams_cache = {}
        self.matches_cache = {}
        self.players_cache = {}
        self.scorers_cache = {}
        self.historical_results = {}
        self.lineups_cache = {}
        self.head2head_cache = {}
        self.standings_cache = {}
        self.weather_cache = {}
        self.team_schedule_cache = {}
        
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
    # GESTION DES LIMITES API (VERSION OPTIMISÉE)
    # ============================================
    
    def _rate_limited_request(self, url, params=None, max_retries=2):
        """Effectue une requête avec gestion des limites de taux (VERSION RAPIDE)"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Réduire la limite à 5 requêtes pour être plus sûr
        if self.request_count >= 5 and time_since_last < 60:
            wait_time = 60 - time_since_last + 1
            print(f"   ⏳ Attente de {wait_time:.1f}s...")
            time.sleep(wait_time)
            self.request_count = 0
            self.last_request_time = time.time()
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=5)  # timeout réduit
                self.request_count += 1
                self.last_request_time = time.time()
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    wait_time = 5  # Réduit de 3-6-9 à 5 secondes fixes
                    print(f"   ⏳ Limite API (429). Attente de {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return response
            except Exception as e:
                print(f"   ⚠️ Erreur requête: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return None
    
    # ============================================
    # MÉTHODES API EXISTANTES (gardées telles quelles)
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
    
    def fetch_historical_results(self, competition_code, months=1):  # Réduit à 1 mois
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
    
    def fetch_upcoming_matches(self, competition_code, days_ahead=3):  # Réduit à 3 jours
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
                for i, match in enumerate(upcoming_matches[:10]):  # Limiter à 10 matchs
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
                'limit': 5  # Réduit à 5 matchs
            }
            
            response = self._rate_limited_request(url, params)
            
            if response and response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                
                if matches:
                    stats = self._calculate_team_stats(matches, team_id)
                    self.teams_cache[cache_key] = stats
                    
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
        
        for match in matches[:3]:  # Limiter à 3 matchs
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
            'form': form[::-1] if form else [3, 3, 3],
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
        weights = [0.4, 0.3, 0.2, 0.1]  # Adapté pour 3-4 matchs
        return sum(f * w for f, w in zip(form_list[:4], weights[:len(form_list)]))
    
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
    # ENTRAÎNEMENT (VERSION ULTRA-OPTIMISÉE)
    # ============================================
    
    def train_models(self):
        """Entraîne les modèles ML avec des données simulées (fallback rapide)"""
        print("\n🤖 Entraînement des modèles (mode fallback rapide)...")
        
        np.random.seed(42)
        n_samples = 200  # Réduit
        n_features = 8   # Réduit
        
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
        """Entraîne les modèles avec des données historiques (VERSION ULTRA-RAPIDE)"""
        print("\n📚 ENTRAÎNEMENT AVEC DONNÉES HISTORIQUES (MODE RAPIDE)")
        print("="*60)
        
        if not competition_codes:
            competition_codes = ['FL1', 'CL']  # Seulement 2 compétitions
        
        all_features = []
        all_results = []
        total_matches = 0
        
        for comp in competition_codes:
            print(f"\n📊 Récupération des données pour {self.competitions[comp]['name']}...")
            matches = self.fetch_historical_results(comp, months=1)  # 1 mois seulement
            
            if matches:
                print(f"   {len(matches)} matchs trouvés")
                total_matches += len(matches)
                
                # Analyser seulement 20 matchs max
                for i, match in enumerate(matches[:20]):
                    try:
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
                    
                    # PAS de pause ici !
        
        print(f"\n📊 Total: {total_matches} matchs analysés, {len(all_features)} échantillons")
        
        if len(all_features) > 10:
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
        """Entraîne le modèle de Poisson (version rapide)"""
        print("\n📊 Entraînement du modèle de Poisson...")
        
        teams_strength = {}
        
        for match in matches[:5]:  # Limiter à 5 matchs
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
        
        for h in range(0, 5):  # Réduit à 0-4 buts
            for a in range(0, 5):
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
        
        top_scores = sorted(score_matrix.items(), key=lambda x: x[1], reverse=True)[:5]
        
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
    # ANALYSE DES MARCHÉS (VERSION SIMPLIFIÉE POUR VITESSE)
    # ============================================
    
    def analyze_all_markets(self, match):
        """Analyse tous les marchés disponibles (version rapide)"""
        print(f"   🔍 Analyse de {match['home_team']} vs {match['away_team']}...")
        
        # Prédiction de base Poisson
        poisson_results = self.predict_poisson_scores(match)
        
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
            'under_25': round(self._calculate_over_under_prob(poisson_results['score_matrix'], 2.5, 'under') * 100, 1),
            'over_25': round(self._calculate_over_under_prob(poisson_results['score_matrix'], 2.5, 'over') * 100, 1),
        }
        
        # 5. SCORES EXACTS
        markets['exact_scores'] = [
            {'score': score, 'probability': round(prob * 100, 1)}
            for score, prob in poisson_results['top_scores'][:3]
        ]
        
        # 6. CONFIDENCE GLOBALE
        markets['global_confidence'] = {
            'average': round(markets['1N2']['confidence'], 1),
            'level': 'HAUTE' if markets['1N2']['confidence'] > 65 else 'MOYENNE'
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
    
    # ============================================
    # IDENTIFICATION DES PARIS SÛRS
    # ============================================
    
    def identify_safe_bets(self, markets):
        """Identifie les éléments les plus sûrs"""
        safe_bets = []
        
        # 1N2
        home_prob = markets['1N2']['home']
        draw_prob = markets['1N2']['draw']
        away_prob = markets['1N2']['away']
        
        if home_prob > 60:
            safe_bets.append({
                'type': '1N2',
                'prediction': 'VICTOIRE DOMICILE',
                'probability': home_prob,
                'confidence': 'HAUTE',
                'emoji': '🏠',
                'reason': f"L'équipe à domicile a {home_prob}% de chances de gagner"
            })
        elif away_prob > 60:
            safe_bets.append({
                'type': '1N2',
                'prediction': 'VICTOIRE EXTERIEUR',
                'probability': away_prob,
                'confidence': 'HAUTE',
                'emoji': '✈️',
                'reason': f"L'équipe à l'extérieur a {away_prob}% de chances de gagner"
            })
        
        # DOUBLE CHANCE
        if markets['double_chance']['1N'] > 80:
            safe_bets.append({
                'type': 'DOUBLE CHANCE',
                'prediction': '1N (Domicile ou Nul)',
                'probability': markets['double_chance']['1N'],
                'confidence': 'TRÈS HAUTE',
                'emoji': '🛡️',
                'reason': "L'équipe à domicile est très solide"
            })
        elif markets['double_chance']['N2'] > 80:
            safe_bets.append({
                'type': 'DOUBLE CHANCE',
                'prediction': 'N2 (Extérieur ou Nul)',
                'probability': markets['double_chance']['N2'],
                'confidence': 'TRÈS HAUTE',
                'emoji': '🛡️',
                'reason': "L'équipe à l'extérieur est très solide"
            })
        
        # BTTS
        btts_oui = markets['btts']['oui']
        btts_non = markets['btts']['non']
        
        if btts_oui > 65:
            safe_bets.append({
                'type': 'BTTS',
                'prediction': 'LES DEUX ÉQUIPES MARQUENT',
                'probability': btts_oui,
                'confidence': 'HAUTE',
                'emoji': '⚽⚽',
                'reason': "Les deux équipes ont une forte capacité offensive"
            })
        elif btts_non > 65:
            safe_bets.append({
                'type': 'BTTS',
                'prediction': 'PAS DE BUT DES DEUX CÔTÉS',
                'probability': btts_non,
                'confidence': 'HAUTE',
                'emoji': '🔒',
                'reason': "Au moins une équipe aura du mal à marquer"
            })
        
        # OVER/UNDER
        over_25 = markets['total_goals']['over_25']
        under_25 = markets['total_goals']['under_25']
        expected = markets['total_goals']['expected']
        
        if over_25 > 65:
            safe_bets.append({
                'type': 'OVER 2.5',
                'prediction': 'PLUS DE 2.5 BUTS',
                'probability': over_25,
                'confidence': 'HAUTE',
                'emoji': '📈',
                'reason': f"Match offensif attendu avec {expected} buts prévus"
            })
        elif under_25 > 65:
            safe_bets.append({
                'type': 'UNDER 2.5',
                'prediction': 'MOINS DE 2.5 BUTS',
                'probability': under_25,
                'confidence': 'HAUTE',
                'emoji': '📉',
                'reason': f"Match fermé attendu avec {expected} buts prévus"
            })
        
        # TRI
        safe_bets.sort(key=lambda x: x['probability'], reverse=True)
        
        return safe_bets
    
    # ============================================
    # SÉLECTION DES MEILLEURS MATCHS
    # ============================================
    
    def select_best_matches(self, matches, max_matches=3):
        """Sélectionne les meilleurs matchs"""
        analyzed = []
        
        for i, match in enumerate(matches[:5]):  # Limiter à 5 matchs
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
# PROGRAMME PRINCIPAL (SIMPLIFIÉ POUR KOYEB)
# ============================================

def main():
    """Point d'entrée principal pour Koyeb"""
    print("="*60)
    print("🏆 FOOTBALL MATCH SELECTOR ULTRA - MODE KOYEB")
    print("="*60)
    
    selector = FootballMatchSelectorUltra(api_key="1b9fa9eead33409cb75f3d0a2df60324")
    
    if selector.test_api_connection():
        print("✅ Connexion API réussie!")
        
        if not selector.is_trained:
            print("\n📚 Entraînement rapide...")
            selector.train_with_historical_data(['FL1', 'CL'])
        
        print("\n" + "="*60)
        print("✅ Initialisation terminée - Prêt pour les requêtes API")
        print("="*60)
        
    else:
        print("❌ Échec connexion API")

if __name__ == "__main__":
    main()