import os
import smtplib
from email.message import EmailMessage


SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")


def send_email(to_email, subject, body):

    if not SMTP_USER or not SMTP_PASS:
        print("\n====== DEV EMAIL ======")
        print("To:", to_email)
        print("Subject:", subject)
        print("Body:", body)
        print("=======================\n")
        return True

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.set_content(body)

    try:

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)

        return True

    except Exception as e:
        print("SMTP error:", e)
        return False