"""SMSEthiopia gateway helper. Auth is KEY header, body is msisdn + text."""

from __future__ import annotations

import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

SMS_URL = "https://smsethiopia.com/api/sms/send"


def normalize_msisdn(phone_number: str) -> str:
    digits = re.sub(r"\D", "", phone_number or "")
    if digits.startswith("251") and len(digits) == 12:
        return digits
    if digits.startswith("0") and len(digits) == 10:
        return "251" + digits[1:]
    if len(digits) == 9 and digits.startswith("9"):
        return "251" + digits
    return digits


def send_sms_notification(phone_number: str, message: str) -> dict:
    api_token = os.getenv("SMS_API_TOKEN")
    if not api_token:
        return {"ok": False, "status": "error", "detail": "SMS_API_TOKEN is missing in .env"}

    clean_phone = normalize_msisdn(phone_number)
    if not re.fullmatch(r"2519\d{8}", clean_phone):
        return {
            "ok": False,
            "status": "error",
            "detail": f"Invalid Ethiopian mobile number: {phone_number}",
        }

    headers = {"KEY": api_token, "Content-Type": "application/json"}
    payload = {"msisdn": clean_phone, "text": message}
    print(f"[SMS] POST {SMS_URL} -> {clean_phone} ({len(message)} chars)")

    try:
        response = requests.post(url=SMS_URL, json=payload, headers=headers, timeout=20)
        print(f"[SMS] HTTP {response.status_code}: {response.text[:400]}")
        try:
            body = response.json()
        except ValueError:
            body = {"raw_response": response.text}

        sent_flag = False
        if isinstance(body, dict):
            sent_flag = bool(body.get("sent") is True or str(body.get("description", "")).lower().startswith("accepted"))
        ok = response.status_code == 200 and (sent_flag or not isinstance(body, dict) or "sent" not in body)
        return {
            "ok": ok,
            "status": "success" if ok else "error",
            "http_status": response.status_code,
            "msisdn": clean_phone,
            "gateway": body,
        }
    except requests.RequestException as exc:
        print(f"[SMS] request failed: {exc}")
        return {"ok": False, "status": "error", "detail": str(exc), "msisdn": clean_phone}
