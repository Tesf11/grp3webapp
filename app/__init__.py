from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables

def create_app():
    app = Flask(__name__)

    # Secret key from .env
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret')

    # Register existing routes (unchanged)
    from .routes import main
    app.register_blueprint(main)

    from .prodcat_api import prodcat_api
    app.register_blueprint(prodcat_api)

    # ✅ Register GenAI routes (new)
    from .routes_genai import genai_bp
    app.register_blueprint(genai_bp)

    return app
