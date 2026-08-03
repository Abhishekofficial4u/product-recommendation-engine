"""
backend/api/auth.py
===================
Authentication utilities — secure password hashing (PBKDF2-HMAC-SHA256)
and native JWT token signature generation and verification.
Zero external library dependencies (eliminates compilation issues).
"""

import os
import hmac
import hashlib
import base64
import json
from datetime import datetime, timedelta
from typing import Optional, Dict

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "rec_engine_super_secret_jwt_key_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# ── Password Cryptography ───────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a random salt."""
    salt = os.urandom(16)
    db_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100000
    )
    return f"{salt.hex()}${db_hash.hex()}"


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against saved salt and hash in constant time."""
    try:
        salt_hex, hash_hex = hashed_password.split("$")
        salt = bytes.fromhex(salt_hex)
        saved_hash = bytes.fromhex(hash_hex)
        
        new_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            100000
        )
        return hmac.compare_digest(new_hash, saved_hash)
    except (ValueError, AttributeError):
        return False


# ── Native JWT Token Engine ─────────────────────────────────────────────────

def base64url_encode(data: bytes) -> str:
    """Encode bytes to base64 URL format, stripping padding '='."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def base64url_decode(payload_str: str) -> bytes:
    """Decode base64 URL format string, restoring padding '='."""
    rem = len(payload_str) % 4
    if rem > 0:
        payload_str += "=" * (4 - rem)
    return base64.urlsafe_b64decode(payload_str.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generate a signed JWT token containing subject and expiry payload."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": int(expire.timestamp())})
    
    # 1. Header & Payload
    header = {"alg": ALGORITHM, "typ": "JWT"}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_json = json.dumps(to_encode, separators=(",", ":")).encode("utf-8")
    
    header_b64 = base64url_encode(header_json)
    payload_b64 = base64url_encode(payload_json)
    
    # 2. Signature
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    sig = hmac.new(
        SECRET_KEY.encode("utf-8"),
        signing_input,
        hashlib.sha256
    ).digest()
    sig_b64 = base64url_encode(sig)
    
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def verify_token(token: str) -> Optional[str]:
    """
    Verify and decode a JWT token.
    Returns the username (sub claim) if valid, or None if expired/corrupted.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        
        header_b64, payload_b64, signature_b64 = parts
        
        # 1. Verify Signature
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(
            SECRET_KEY.encode("utf-8"),
            signing_input,
            hashlib.sha256
        ).digest()
        
        provided_sig = base64url_decode(signature_b64)
        if not hmac.compare_digest(provided_sig, expected_sig):
            return None
        
        # 2. Parse and Validate Payload
        payload_bytes = base64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
        
        exp = payload.get("exp")
        if exp is None:
            return None
        
        # Check Expiry
        if datetime.utcnow().timestamp() > exp:
            return None
            
        return payload.get("sub")
    except Exception:
        return None
