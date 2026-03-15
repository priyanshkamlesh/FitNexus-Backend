import secrets

OTP_LENGTH = 6


def generate_otp():

    return "".join(
        secrets.choice("0123456789")
        for _ in range(OTP_LENGTH)
    )