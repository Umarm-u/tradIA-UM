"""
tradIA Auth Server — auth_app.py
=================================
Handles user registration, login, sessions, and serves the authenticated
app shell wrapping the existing dashboard (port 5000).

Runs on port 5001. dashboard.py continues to run on port 5000 untouched.
"""
import os
import urllib.parse
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, session,
    redirect, jsonify, send_from_directory,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet

load_dotenv()

BASE_DIR     = Path(__file__).parent
INSTANCE_DIR   = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(exist_ok=True)
DASHBOARD_URL  = os.environ.get("DASHBOARD_URL", "http://localhost:5000")

app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ["AUTH_SECRET_KEY"]
app.config.update(
    SQLALCHEMY_DATABASE_URI  = f"sqlite:///{INSTANCE_DIR / 'users.db'}",
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    SESSION_COOKIE_HTTPONLY  = True,
    SESSION_COOKIE_SAMESITE  = "Lax",
)

db      = SQLAlchemy(app)
_fernet = Fernet(os.environ["FERNET_KEY"].encode())


# ── Model ─────────────────────────────────────────────────────────────────────

class User(db.Model):
    id              = db.Column(db.Integer,     primary_key=True)
    name            = db.Column(db.String(120),  nullable=False)
    email           = db.Column(db.String(255),  unique=True, nullable=False)
    password_hash   = db.Column(db.String(255),  nullable=False)
    binance_api_key = db.Column(db.Text,         nullable=True)   # Fernet-encrypted
    binance_secret  = db.Column(db.Text,         nullable=True)   # Fernet-encrypted
    created_at      = db.Column(db.DateTime,
                        default=lambda: datetime.now(timezone.utc))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enc(value: str) -> str:
    return _fernet.encrypt(value.encode()).decode()

def _dec(token: str) -> str:
    return _fernet.decrypt(token.encode()).decode()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect("/")
        return f(*args, **kwargs)
    return decorated


# ── Static / landing ──────────────────────────────────────────────────────────

@app.route("/")
def landing():
    if "user_id" in session:
        return redirect("/app")
    error = request.args.get("e", "")
    # Pass error flag via cookie so the static landing page can read it
    resp = send_from_directory("landing", "index.html")
    if error:
        resp.set_cookie("auth_error", error, max_age=10, samesite="Lax")
    return resp

@app.route("/logo2.svg")
def logo():
    return send_from_directory("dashboard", "logo2.svg")


# ── Auth API ──────────────────────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def api_register():
    data  = request.get_json(silent=True) or {}
    name  = (data.get("name")     or "").strip()
    email = (data.get("email")    or "").strip().lower()
    pw    = (data.get("password") or "")

    if not name or not email or not pw:
        return jsonify(ok=False, error="All fields are required."), 400
    if len(pw) < 8:
        return jsonify(ok=False, error="Password must be at least 8 characters."), 400
    if User.query.filter_by(email=email).first():
        return jsonify(ok=False, error="An account with this email already exists."), 409

    user = User(
        name          = name,
        email         = email,
        password_hash = generate_password_hash(pw),
    )
    db.session.add(user)
    db.session.commit()

    session["user_id"]    = user.id
    session["user_name"]  = user.name
    session["user_email"] = user.email
    return jsonify(ok=True, redirect="/app")


@app.route("/api/login", methods=["POST"])
def api_login():
    data  = request.get_json(silent=True) or {}
    email = (data.get("email")    or "").strip().lower()
    pw    = (data.get("password") or "")

    if not email or not pw:
        return jsonify(ok=False, error="Email and password are required."), 400

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, pw):
        return jsonify(ok=False, error="Invalid email or password."), 401

    session["user_id"]    = user.id
    session["user_name"]  = user.name
    session["user_email"] = user.email
    return jsonify(ok=True, redirect="/app")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ── Protected routes ──────────────────────────────────────────────────────────

@app.route("/app")
@login_required
def app_shell():
    return render_template("app_shell.html",
        user_name     = session.get("user_name",  ""),
        user_email    = session.get("user_email", ""),
        dashboard_url = DASHBOARD_URL,
    )


@app.route("/account")
@login_required
def account_page():
    user = db.session.get(User, session["user_id"])
    return render_template("account.html",
        user      = user,
        has_keys  = bool(user.binance_api_key),
        user_name = session.get("user_name", ""),
    )


@app.route("/api/account", methods=["POST"])
@login_required
def update_account():
    user = db.session.get(User, session["user_id"])
    data = request.get_json(silent=True) or {}

    name = (data.get("name") or "").strip()
    if name:
        user.name = name
        session["user_name"] = name

    api_key = (data.get("binance_api_key") or "").strip()
    secret  = (data.get("binance_secret")  or "").strip()
    if api_key and secret:
        user.binance_api_key = _enc(api_key)
        user.binance_secret  = _enc(secret)

    db.session.commit()
    return jsonify(ok=True, message="Account updated.")


# ── Google OAuth (requires GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET in .env) ───

@app.route("/auth/google")
def google_login():
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    if not client_id:
        return redirect("/?e=google")
    params = {
        "client_id":     client_id,
        "redirect_uri":  request.url_root.rstrip("/") + "/auth/google/callback",
        "response_type": "code",
        "scope":         "openid email profile",
        "prompt":        "select_account",
    }
    return redirect(
        "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    )


@app.route("/auth/google/callback")
def google_callback():
    import requests as _req
    code          = request.args.get("code")
    client_id     = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    if not code or not client_id or not client_secret:
        return redirect("/?e=google")

    token_resp = _req.post("https://oauth2.googleapis.com/token", data={
        "code":          code,
        "client_id":     client_id,
        "client_secret": client_secret,
        "redirect_uri":  request.url_root.rstrip("/") + "/auth/google/callback",
        "grant_type":    "authorization_code",
    }).json()

    access_token = token_resp.get("access_token")
    if not access_token:
        return redirect("/?e=google")

    info = _req.get("https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"}
    ).json()

    email = (info.get("email") or "").lower()
    name  = info.get("name") or email.split("@")[0]
    if not email:
        return redirect("/?e=google")

    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(
            name          = name,
            email         = email,
            password_hash = generate_password_hash(os.urandom(32).hex()),
        )
        db.session.add(user)
        db.session.commit()

    session["user_id"]    = user.id
    session["user_name"]  = user.name
    session["user_email"] = user.email
    return redirect("/app")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5001, debug=False)
