from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import football_selector
import json
from datetime import datetime
import os
import logging
import time
import threading

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = 'votre_cle_secrete_ultra_123456789'
CORS(app)

print("🚀 Initialisation du programme ULTRA INTELLIGENT...")

# Récupérer la clé API depuis les variables d'environnement (plus sécurisé)
API_KEY = os.environ.get('API_KEY', "1b9fa9eead33409cb75f3d0a2df60324")
selector = football_selector.FootballMatchSelectorUltra(api_key=API_KEY)

# Variable pour suivre l'état d'entraînement
training_status = {
    'in_progress': False,
    'progress': 0,
    'message': '',
    'completed': False
}

def train_in_background():
    """Entraîne les modèles en arrière-plan de façon progressive"""
    global training_status
    training_status['in_progress'] = True
    training_status['message'] = 'Récupération des données...'
    
    try:
        # Entraînement limité pour éviter les timeouts
        competitions_to_train = ['FL1', 'CL']  # Seulement Ligue 1 et Champions League
        selector.train_with_historical_data(competitions_to_train)
        
        training_status['message'] = 'Entraînement terminé avec succès!'
        training_status['completed'] = True
    except Exception as e:
        training_status['message'] = f'Erreur: {str(e)}'
    finally:
        training_status['in_progress'] = False

# Lancer l'entraînement en arrière-plan seulement si nécessaire
if not selector.is_trained:
    print("\n📚 Entraînement en arrière-plan...")
    thread = threading.Thread(target=train_in_background)
    thread.daemon = True
    thread.start()
else:
    print("✅ Modèles déjà entraînés chargés!")
    training_status['completed'] = True

print("✅ Programme initialisé!")

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/predictions')
def predictions():
    """Page des prédictions"""
    return render_template('predictions.html', competitions=selector.competitions)

@app.route('/api/training-status', methods=['GET'])
def get_training_status():
    """Retourne le statut de l'entraînement"""
    return jsonify({
        'success': True,
        'is_trained': selector.is_trained or training_status['completed'],
        'training_in_progress': training_status['in_progress'],
        'message': training_status['message']
    })

@app.route('/api/competitions', methods=['GET'])
def get_competitions():
    """Liste des compétitions disponibles"""
    return jsonify({
        'success': True,
        'competitions': selector.competitions
    })

@app.route('/api/matches', methods=['POST'])
def get_matches():
    """Récupère les matchs sans analyse poussée (rapide)"""
    data = request.json
    competition_code = data.get('competition')
    days_ahead = int(data.get('days_ahead', 3))  # Réduit à 3 jours par défaut
    
    if not competition_code:
        return jsonify({'success': False, 'error': 'Code compétition manquant'}), 400
    
    try:
        # Réinitialiser le compteur de requêtes
        selector.request_count = 0
        
        # Récupérer les matchs
        matches = selector.fetch_upcoming_matches(competition_code, days_ahead)
        
        formatted_matches = []
        for match in matches:
            formatted_matches.append({
                'id': match['id'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'date': match['date'],
                'time': match.get('time', '--:--'),
                'competition': match['competition_name']
            })
        
        return jsonify({
            'success': True,
            'matches': formatted_matches,
            'count': len(formatted_matches)
        })
        
    except Exception as e:
        logging.error(f"Erreur récupération matchs: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-match', methods=['POST'])
def analyze_match():
    """Analyse UN SEUL match à la fois (rapide)"""
    data = request.json
    match_id = data.get('match_id')
    
    if not match_id:
        return jsonify({'success': False, 'error': 'ID match manquant'}), 400
    
    try:
        # Chercher le match dans les compétitions récentes
        matches = selector.fetch_upcoming_matches('FL1', 3)  # Récupérer quelques matchs
        target_match = None
        
        for match in matches:
            if str(match['id']) == str(match_id):
                target_match = match
                break
        
        if not target_match:
            return jsonify({'success': False, 'error': 'Match non trouvé'})
        
        # Analyser UNIQUEMENT ce match
        markets = selector.analyze_all_markets(target_match)
        safe_bets = selector.identify_safe_bets(markets)
        
        return jsonify({
            'success': True,
            'match': {
                'home_team': target_match['home_team'],
                'away_team': target_match['away_team'],
                'date': target_match['date'],
                'time': target_match.get('time', '--:--'),
                'competition': target_match['competition_name']
            },
            'markets': markets,
            'safe_bets': safe_bets[:5]  # Top 5 paris sûrs
        })
        
    except Exception as e:
        logging.error(f"Erreur analyse match: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    """Endpoint de health check pour Koyeb"""
    return jsonify({'status': 'healthy', 'trained': selector.is_trained})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))  # Koyeb utilise le port 8000
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("\n" + "="*60)
    print("🚀 CASHGOAL ULTRA INTELLIGENT - Serveur démarré")
    print(f"📍 http://0.0.0.0:{port}")
    print("🤖 Mode: Prédictions intelligentes optimisées pour Koyeb")
    print("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)