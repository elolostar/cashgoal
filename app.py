from flask import Flask, render_template, request, jsonify
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
selector = football_selector.FootballMatchSelectorUltra(api_key="1b9fa9eead33409cb75f3d0a2df60324")

# Variable pour suivre l'état d'entraînement
training_status = {
    'in_progress': False,
    'progress': 0,
    'message': ''
}

def train_in_background():
    """Entraîne les modèles en arrière-plan"""
    global training_status
    training_status['in_progress'] = True
    training_status['message'] = 'Récupération des données...'
    
    try:
        selector.train_with_historical_data()
        training_status['message'] = 'Entraînement terminé avec succès!'
    except Exception as e:
        training_status['message'] = f'Erreur: {str(e)}'
    finally:
        training_status['in_progress'] = False

# Lancer l'entraînement en arrière-plan
if not selector.is_trained:
    print("\n📚 Entraînement en arrière-plan...")
    thread = threading.Thread(target=train_in_background)
    thread.daemon = True
    thread.start()

print("✅ Programme initialisé!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictions')
def predictions():
    return render_template('predictions.html', competitions=selector.competitions)

@app.route('/api/training-status', methods=['GET'])
def get_training_status():
    """Retourne le statut de l'entraînement"""
    return jsonify({
        'success': True,
        'is_trained': selector.is_trained,
        'training_in_progress': training_status['in_progress'],
        'message': training_status['message']
    })

@app.route('/api/competitions', methods=['GET'])
def get_competitions():
    return jsonify({
        'success': True,
        'competitions': selector.competitions
    })

@app.route('/api/todays-matches', methods=['POST'])
def get_todays_matches():
    data = request.json
    competition_code = data.get('competition')
    days_ahead = int(data.get('days_ahead', 0))
    
    if not competition_code:
        return jsonify({'success': False, 'error': 'Code compétition manquant'}), 400
    
    try:
        selector.request_count = 0
        
        if days_ahead == 0:
            matches = selector.fetch_todays_matches(competition_code)
        else:
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
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-matches', methods=['POST'])
def analyze_matches():
    data = request.json
    competition_code = data.get('competition')
    days_ahead = int(data.get('days_ahead', 7))
    n_matches = int(data.get('n_matches', 5))
    
    if not competition_code:
        return jsonify({'success': False, 'error': 'Code compétition manquant'}), 400
    
    try:
        selector.request_count = 0
        
        if days_ahead == 0:
            matches = selector.fetch_todays_matches(competition_code)
        else:
            matches = selector.fetch_upcoming_matches(competition_code, days_ahead)
        
        if not matches:
            return jsonify({'success': False, 'error': 'Aucun match trouvé'})
        
        # Poisson model training
        selector.train_poisson_model(matches)
        
        best_matches = selector.select_best_matches(matches, n_matches)
        
        results = []
        for result in best_matches:
            match = result['match']
            markets = result['markets']
            
            safe_bets = selector.identify_safe_bets(markets)
            
            formatted = {
                'match': {
                    'id': match['id'],
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'date': match['date'],
                    'time': match.get('time', '--:--'),
                    'competition': match['competition_name']
                },
                'confidence': result['confidence'],
                'markets': markets,
                'safe_bets': safe_bets
            }
            results.append(formatted)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 CASHGOAL ULTRA INTELLIGENT - Serveur démarré")
    print("📍 http://localhost:5000")
    print("🤖 Mode: Prédictions intelligentes avec identification des paris sûrs")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)