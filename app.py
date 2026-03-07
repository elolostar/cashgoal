from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import football_selector
import json
from datetime import datetime
import os
import logging
import time
import threading
import traceback

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY', 'votre_cle_secrete_ultra_123456789')
CORS(app)

print("="*60)
print("🚀 CASHGOAL ULTRA INTELLIGENT - DÉMARRAGE")
print("="*60)

# Récupérer la clé API depuis les variables d'environnement (plus sécurisé)
API_KEY = os.environ.get('API_KEY', "1b9fa9eead33409cb75f3d0a2df60324")
print(f"🔑 Clé API: {API_KEY[:5]}...{API_KEY[-5:]}")

# Initialisation du sélecteur
print("🔄 Initialisation du FootballMatchSelectorUltra...")
selector = football_selector.FootballMatchSelectorUltra(api_key=API_KEY)

# Variable pour suivre l'état d'entraînement
training_status = {
    'in_progress': False,
    'progress': 0,
    'message': '',
    'completed': False,
    'start_time': None,
    'end_time': None
}

def train_in_background():
    """Entraîne les modèles en arrière-plan de façon progressive"""
    global training_status
    training_status['in_progress'] = True
    training_status['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    training_status['message'] = 'Récupération des données...'
    
    try:
        print("\n📚 Entraînement en arrière-plan...")
        # Entraînement limité pour éviter les timeouts
        competitions_to_train = ['FL1', 'CL']  # Seulement Ligue 1 et Champions League
        selector.train_with_historical_data(competitions_to_train)
        
        training_status['message'] = '✅ Entraînement terminé avec succès!'
        training_status['completed'] = True
        training_status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("✅ Entraînement terminé!")
        
    except Exception as e:
        error_msg = f"❌ Erreur entraînement: {str(e)}"
        training_status['message'] = error_msg
        print(error_msg)
        traceback.print_exc()
    finally:
        training_status['in_progress'] = False

# Lancer l'entraînement en arrière-plan seulement si nécessaire
if not selector.is_trained:
    print("\n📚 Modèles non entraînés - Lancement de l'entraînement en arrière-plan...")
    thread = threading.Thread(target=train_in_background)
    thread.daemon = True
    thread.start()
else:
    print("✅ Modèles déjà entraînés chargés!")
    training_status['completed'] = True
    training_status['message'] = '✅ Modèles pré-entraînés chargés'

print("✅ Programme initialisé avec succès!")
print("="*60)

# ============================================
# ROUTES PRINCIPALES
# ============================================

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/predictions')
def predictions():
    """Page des prédictions"""
    return render_template('predictions.html', competitions=selector.competitions)

@app.route('/health')
def health():
    """Endpoint de health check pour Koyeb"""
    return jsonify({
        'status': 'healthy',
        'trained': selector.is_trained or training_status['completed'],
        'training': training_status['in_progress'],
        'timestamp': datetime.now().isoformat()
    })

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/training-status', methods=['GET'])
def get_training_status():
    """Retourne le statut de l'entraînement"""
    return jsonify({
        'success': True,
        'is_trained': selector.is_trained or training_status['completed'],
        'training_in_progress': training_status['in_progress'],
        'message': training_status['message'],
        'start_time': training_status['start_time'],
        'end_time': training_status['end_time']
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
    days_ahead = int(data.get('days_ahead', 3))  # 3 jours par défaut
    
    if not competition_code:
        return jsonify({'success': False, 'error': 'Code compétition manquant'}), 400
    
    try:
        logger.info(f"Récupération matchs pour {competition_code} (J+{days_ahead})")
        
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
        
        logger.info(f"✅ {len(formatted_matches)} matchs trouvés")
        
        return jsonify({
            'success': True,
            'matches': formatted_matches,
            'count': len(formatted_matches)
        })
        
    except Exception as e:
        logger.error(f"Erreur récupération matchs: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-match', methods=['POST'])
@app.route('/api/analyze-matches', methods=['POST'])  # Alias pour compatibilité
def analyze_match():
    """Analyse UN SEUL match à la fois"""
    data = request.json
    match_id = data.get('match_id')
    
    if not match_id:
        return jsonify({'success': False, 'error': 'ID match manquant'}), 400
    
    try:
        logger.info(f"🔍 Analyse du match {match_id}")
        
        # Chercher le match dans les compétitions récentes
        matches = selector.fetch_upcoming_matches('FL1', 3)
        if not matches:
            matches = selector.fetch_upcoming_matches('CL', 3)
        
        target_match = None
        for match in matches:
            if str(match['id']) == str(match_id):
                target_match = match
                break
        
        if not target_match:
            return jsonify({'success': False, 'error': 'Match non trouvé'}), 404
        
        # Analyser UNIQUEMENT ce match
        markets = selector.analyze_all_markets(target_match)
        safe_bets = selector.identify_safe_bets(markets)
        
        logger.info(f"✅ Analyse terminée pour {target_match['home_team']} vs {target_match['away_team']}")
        
        return jsonify({
            'success': True,
            'match': {
                'id': target_match['id'],
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
        logger.error(f"Erreur analyse match: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-competition', methods=['POST'])
def analyze_competition():
    """Analyse plusieurs matchs d'une compétition"""
    data = request.json
    competition_code = data.get('competition')
    days_ahead = int(data.get('days_ahead', 3))
    max_matches = int(data.get('max_matches', 3))
    
    if not competition_code:
        return jsonify({'success': False, 'error': 'Code compétition manquant'}), 400
    
    try:
        logger.info(f"🔍 Analyse compétition {competition_code} (J+{days_ahead})")
        
        # Récupérer les matchs
        matches = selector.fetch_upcoming_matches(competition_code, days_ahead)
        
        if not matches:
            return jsonify({'success': False, 'error': 'Aucun match trouvé'}), 404
        
        # Entraîner le modèle Poisson
        selector.train_poisson_model(matches)
        
        # Sélectionner les meilleurs matchs
        best_matches = selector.select_best_matches(matches, max_matches)
        
        results = []
        for result in best_matches:
            match = result['match']
            markets = result['markets']
            safe_bets = selector.identify_safe_bets(markets)
            
            results.append({
                'match': {
                    'id': match['id'],
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'date': match['date'],
                    'time': match.get('time', '--:--'),
                    'competition': match['competition_name']
                },
                'confidence': result['confidence'],
                'markets': {
                    '1N2': markets['1N2'],
                    'double_chance': markets['double_chance'],
                    'btts': markets['btts'],
                    'total_goals': markets['total_goals'],
                    'exact_scores': markets.get('exact_scores', [])[:3]
                },
                'safe_bets': safe_bets[:3]
            })
        
        logger.info(f"✅ {len(results)} matchs analysés")
        
        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Erreur analyse compétition: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Endpoint de débogage"""
    return jsonify({
        'success': True,
        'api_key_configured': bool(API_KEY),
        'api_key_preview': API_KEY[:5] + '...' + API_KEY[-5:] if API_KEY else None,
        'is_trained': selector.is_trained,
        'training_status': training_status,
        'competitions_count': len(selector.competitions),
        'cache_stats': {
            'teams': len(selector.teams_cache),
            'lineups': len(selector.lineups_cache),
            'head2head': len(selector.head2head_cache)
        }
    })

# ============================================
# GESTION DES ERREURS
# ============================================

@app.errorhandler(404)
def not_found(error):
    """Page 404 personnalisée"""
    return jsonify({'success': False, 'error': 'Endpoint non trouvé'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Erreur serveur"""
    return jsonify({'success': False, 'error': 'Erreur interne du serveur'}), 500

# ============================================
# LANCEMENT DU SERVEUR
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))  # Koyeb utilise le port 8000
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("\n" + "="*60)
    print("🚀 CASHGOAL ULTRA INTELLIGENT - SERVEUR DÉMARRÉ")
    print(f"📍 http://0.0.0.0:{port}")
    print(f"🔧 Mode debug: {debug}")
    print("📊 Endpoints disponibles:")
    print("   - GET  /                    Page d'accueil")
    print("   - GET  /predictions          Page des prédictions")
    print("   - GET  /health               Health check")
    print("   - GET  /api/training-status   Statut entraînement")
    print("   - GET  /api/competitions      Liste compétitions")
    print("   - POST /api/matches           Récupérer matchs")
    print("   - POST /api/analyze-match     Analyser un match")
    print("   - POST /api/analyze-competition Analyser compétition")
    print("   - GET  /api/debug             Infos débogage")
    print("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)