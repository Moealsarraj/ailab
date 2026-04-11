"""AI Lab — Flask app factory."""
import os
from flask import Flask
from flask_wtf.csrf import CSRFProtect

_csrf = CSRFProtect()


def create_app():
    app = Flask(__name__, template_folder="templates")
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-ailab-2026")
    app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
    _csrf.init_app(app)

    from app.home.routes import bp as home_bp
    app.register_blueprint(home_bp)

    from app.tools.prompt_bench.routes import bp as prompt_bench_bp
    app.register_blueprint(prompt_bench_bp, url_prefix='/prompt-bench')

    from app.tools.prompt_shield.routes import bp as prompt_shield_bp
    app.register_blueprint(prompt_shield_bp, url_prefix='/prompt-shield')

    from app.tools.agent_builder.routes import bp as agent_builder_bp
    app.register_blueprint(agent_builder_bp, url_prefix='/agent-builder')

    from app.tools.arabic_bench.routes import bp as arabic_bench_bp
    app.register_blueprint(arabic_bench_bp, url_prefix='/arabic-bench')

    from flask import jsonify
    @app.errorhandler(Exception)
    def _handle_exc(e):
        code = getattr(e, "code", 500)
        return jsonify({"error": str(e)}), code

    return app