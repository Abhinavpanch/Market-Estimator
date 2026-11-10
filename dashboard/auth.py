import hashlib
import hmac
import os
import re
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

from flask import Blueprint, current_app, jsonify, request, session
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from werkzeug.security import check_password_hash, generate_password_hash

auth_api = Blueprint("auth_api", __name__)
_mongo_client = None
_indexes_ready = False
_DUMMY_PASSWORD_HASH = generate_password_hash("not-a-real-user-password")
_TICKER_PATTERN = re.compile(r"[A-Z0-9.^=-]{1,20}")
_EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class MailDeliveryError(Exception):
    pass


def get_users():
    global _mongo_client, _indexes_ready
    uri = os.environ.get("MONGO_URI")
    if not uri:
        raise RuntimeError("Set MONGO_URI to enable account and watchlist storage.")
    if not _mongo_client:
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000, tz_aware=True)
    database = _mongo_client[os.environ.get("MONGO_DB_NAME", "ai_stock_estimator")]
    users = database.users
    if not _indexes_ready:
        users.create_index("username_key", unique=True)
        users.create_index("email_key", unique=True)
        _indexes_ready = True
    return users


def get_user(user_id):
    return get_users().find_one({"_id": user_id, "verified": True})


def _send_verification_email(email, username, code):
    host = os.environ.get("SMTP_HOST")
    sender = os.environ.get("MAIL_FROM")
    if not host or not sender:
        raise RuntimeError("Set SMTP_HOST and MAIL_FROM to send verification emails.")

    message = EmailMessage()
    message["Subject"] = "Verify your AI Stock Estimator account"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        f"Hi {username},\n\nYour verification code is {code}. "
        "It expires in 10 minutes.\n"
    )

    try:
        port = int(os.environ.get("SMTP_PORT", "587"))
    except ValueError as exc:
        raise RuntimeError("SMTP_PORT must be a valid port number.") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("SMTP_PORT must be between 1 and 65535.")
    use_ssl = os.environ.get("SMTP_USE_SSL", "").lower() in {"1", "true", "yes"}
    smtp_class = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
    try:
        with smtp_class(host, port, timeout=15) as smtp:
            if not use_ssl:
                smtp.starttls()
            username_config = os.environ.get("SMTP_USERNAME")
            if username_config:
                smtp.login(username_config, os.environ.get("SMTP_PASSWORD", ""))
            smtp.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        raise MailDeliveryError("Could not send the verification email.") from exc


def _otp_hash(code):
    secret = current_app.secret_key
    if not secret:
        raise RuntimeError("Set FLASK_SECRET_KEY before using account features.")
    return hmac.new(str(secret).encode(), code.encode(), hashlib.sha256).hexdigest()


def _json_body():
    return request.get_json(silent=True) or {}


def _valid_signup(username, email, password):
    return (
        re.fullmatch(r"[A-Za-z0-9_]{3,32}", username) is not None
        and len(email) <= 254
        and _EMAIL_PATTERN.fullmatch(email) is not None
        and 10 <= len(password) <= 128
    )


@auth_api.post("/api/auth/signup")
def signup():
    body = _json_body()
    username = str(body.get("username", "")).strip()
    email = str(body.get("email", "")).strip().lower()
    password = body.get("password", "")
    if not isinstance(password, str) or not _valid_signup(username, email, password):
        return jsonify({
            "error": "Use a 3–32 character username, valid email, and password of 10–128 characters."
        }), 400

    users = get_users()
    username_key = username.lower()
    existing = users.find_one({
        "$or": [{"username_key": username_key}, {"email_key": email}]
    })
    if existing:
        if existing.get("verified"):
            return jsonify({"error": "Username or email is already registered."}), 409
        if existing["username_key"] != username_key or existing["email_key"] != email:
            return jsonify({"error": "Username or email is already registered."}), 409
        sent_at = existing.get("otp_sent_at")
        if sent_at and sent_at > datetime.now(timezone.utc) - timedelta(seconds=60):
            return jsonify({"error": "Please wait before requesting another code."}), 429

    code = f"{secrets.randbelow(1_000_000):06d}"
    now = datetime.now(timezone.utc)
    record = {
        "username": username,
        "username_key": username_key,
        "email": email,
        "email_key": email,
        "password_hash": generate_password_hash(password),
        "verified": False,
        "otp_hash": _otp_hash(code),
        "otp_expires_at": now + timedelta(minutes=10),
        "otp_sent_at": now,
        "otp_attempts": 0,
        "watchlist": [],
        "created_at": now,
    }

    try:
        if existing:
            users.update_one({"_id": existing["_id"]}, {"$set": record})
        else:
            record["_id"] = secrets.token_urlsafe(24)
            users.insert_one(record)
    except DuplicateKeyError:
        return jsonify({"error": "Username or email is already registered."}), 409

    try:
        _send_verification_email(email, username, code)
    except (MailDeliveryError, RuntimeError) as exc:
        users.update_one(
            {"_id": record.get("_id", existing["_id"] if existing else None)},
            {"$unset": {"otp_hash": "", "otp_expires_at": "", "otp_sent_at": ""}},
        )
        current_app.logger.error("Verification email delivery failed: %s", exc)
        return jsonify({"error": str(exc)}), 503
    return jsonify({"message": "Check your email for a verification code."}), 201


@auth_api.post("/api/auth/verify")
def verify_email():
    body = _json_body()
    email = str(body.get("email", "")).strip().lower()
    code = str(body.get("code", "")).strip()
    if not _EMAIL_PATTERN.fullmatch(email) or not re.fullmatch(r"\d{6}", code):
        return jsonify({"error": "Enter a valid email and six-digit code."}), 400

    users = get_users()
    user = users.find_one({"email_key": email, "verified": False})
    now = datetime.now(timezone.utc)
    if (
        not user
        or user.get("otp_attempts", 0) >= 5
        or not user.get("otp_expires_at")
        or user["otp_expires_at"] <= now
        or not hmac.compare_digest(user.get("otp_hash", ""), _otp_hash(code))
    ):
        if user and user.get("otp_expires_at") and user["otp_expires_at"] > now:
            users.update_one({"_id": user["_id"]}, {"$inc": {"otp_attempts": 1}})
        return jsonify({"error": "Invalid or expired verification code."}), 400

    users.update_one(
        {"_id": user["_id"]},
        {"$set": {"verified": True}, "$unset": {
            "otp_hash": "", "otp_expires_at": "", "otp_sent_at": "", "otp_attempts": ""
        }},
    )
    return jsonify({"message": "Email verified. You can now log in."})


@auth_api.post("/api/auth/login")
def login():
    body = _json_body()
    identifier = str(body.get("identifier", "")).strip()
    password = body.get("password", "")
    if not identifier or not isinstance(password, str) or len(password) > 128:
        return jsonify({"error": "Enter your username or email and password."}), 400

    key = identifier.lower()
    user = get_users().find_one({
        "$or": [{"username_key": key}, {"email_key": key}]
    })
    if user and user.get(
        "login_locked_until", datetime.min.replace(tzinfo=timezone.utc)
    ) > datetime.now(timezone.utc):
        return jsonify({"error": "Too many failed attempts. Try again later."}), 429
    password_hash = user["password_hash"] if user else _DUMMY_PASSWORD_HASH
    valid_password = check_password_hash(password_hash, password)
    if not user or not valid_password:
        if user:
            get_users().update_one({"_id": user["_id"]}, {"$inc": {"login_attempts": 1}})
            get_users().update_one(
                {"_id": user["_id"], "login_attempts": {"$gte": 5}},
                {"$set": {"login_locked_until": datetime.now(timezone.utc) + timedelta(minutes=10)}},
            )
        return jsonify({"error": "Invalid username/email or password."}), 401
    if not user.get("verified"):
        return jsonify({"error": "Verify your email before logging in."}), 403

    session.clear()
    session.permanent = True
    session["user_id"] = user["_id"]
    session["csrf_token"] = secrets.token_urlsafe(32)
    get_users().update_one(
        {"_id": user["_id"]},
        {"$unset": {"login_attempts": "", "login_locked_until": ""}},
    )
    return jsonify({
        "username": user["username"],
        "csrfToken": session["csrf_token"],
    })


@auth_api.get("/api/auth/me")
def current_session():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"authenticated": False})
    user = get_user(user_id)
    if not user:
        session.clear()
        return jsonify({"authenticated": False})
    if not session.get("csrf_token"):
        session["csrf_token"] = secrets.token_urlsafe(32)
    return jsonify({
        "authenticated": True,
        "username": user["username"],
        "csrfToken": session["csrf_token"],
    })


@auth_api.post("/api/auth/logout")
def logout():
    session.clear()
    return jsonify({"message": "Logged out."})


@auth_api.get("/api/watchlist")
def get_watchlist():
    user = get_user(session["user_id"])
    return jsonify({"tickers": user.get("watchlist", [])})


@auth_api.post("/api/watchlist")
def add_to_watchlist():
    ticker = str(_json_body().get("ticker", "")).strip().upper()
    if not _TICKER_PATTERN.fullmatch(ticker):
        return jsonify({"error": "Enter a valid stock ticker symbol."}), 400

    users = get_users()
    user = users.find_one({"_id": session["user_id"], "verified": True})
    tickers = user.get("watchlist", [])
    if ticker not in tickers and len(tickers) >= 100:
        return jsonify({"error": "Watchlists are limited to 100 stocks."}), 400
    users.update_one(
        {"_id": user["_id"]},
        {"$addToSet": {"watchlist": ticker}},
    )
    return jsonify({"ticker": ticker, "tickers": sorted(set(tickers) | {ticker})})


@auth_api.delete("/api/watchlist/<ticker>")
def remove_from_watchlist(ticker):
    users = get_users()
    user = users.find_one({"_id": session["user_id"], "verified": True})
    users.update_one({"_id": user["_id"]}, {"$pull": {"watchlist": ticker.upper()}})
    return jsonify({
        "tickers": sorted(set(user.get("watchlist", [])) - {ticker.upper()})
    })


def register_auth_routes(app):
    app.register_blueprint(auth_api)

    @app.after_request
    def set_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response
