import os
from pathlib import Path
_env = Path(__file__).parent / ".env"
if _env.exists():
    from dotenv import load_dotenv
    load_dotenv(_env, override=True)
from app import create_app
app = create_app()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7862, debug=False)
