# run.py
from app import app

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     🚀 CASHGOAL - APPLICATION DE PRÉDICTION DE MATCHS       ║
    ║                                                              ║
    ║     📍 Serveur démarré sur: http://localhost:5000          ║
    ║     📍 Pour arrêter: Ctrl+C                                 ║
    ║                                                              ║
    ║     ⚠️  Les paris sportifs comportent des risques           ║
    ║        Jouez responsablement !                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)