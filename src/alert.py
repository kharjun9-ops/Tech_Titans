import os
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def _build_media_url(clip_path: str, base_url: str) -> str:
    if not clip_path or not base_url:
        return ""
    clip_name = os.path.basename(clip_path)
    url = f"{base_url.rstrip('/')}/{clip_name}"
    # ngrok free tunnels show a browser warning unless this param is present.
    if "ngrok" in base_url:
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query))
        query.setdefault("ngrok-skip-browser-warning", "1")
        url = urlunparse(parsed._replace(query=urlencode(query)))
    return url


def send_alert(reason: str, clip_path: str, config) -> None:
    body = f"ALERT: {reason}"
    clip_url = _build_media_url(clip_path, config.public_clip_base_url)

    if config.send_whatsapp and config.twilio_account_sid and config.twilio_auth_token:
        try:
            from twilio.rest import Client

            client = Client(config.twilio_account_sid, config.twilio_auth_token)
            kwargs = {}
            if clip_url:
                kwargs["media_url"] = [clip_url]
            client.messages.create(
                from_=config.whatsapp_from,
                to=config.whatsapp_to,
                body=body,
                **kwargs,
            )
            print("WhatsApp alert sent.")
            return
        except Exception as exc:
            print(f"WhatsApp send failed: {exc}")

    if clip_path:
        print(f"SIMULATED ALERT: {body} | clip={clip_path}")
    else:
        print(f"SIMULATED ALERT: {body}")
