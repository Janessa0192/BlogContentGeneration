import re
import config


class ProviderError(Exception):
    """Base class. Messages are always safe to print (keys redacted)."""

class ConfigError(ProviderError): pass
class InvalidKeyError(ProviderError): pass
class RateLimitError(ProviderError): pass
class QuotaExhaustedError(ProviderError): pass


def redact(text) -> str:
    s = str(text)
    for secret in config.secrets():
        if secret:
            s = s.replace(secret, "***")
    s = re.sub(r"(key=)[^&\s'\"]+", r"\1***", s)
    s = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "***", s)
    s = re.sub(r"tvly-[0-9A-Za-z_\-]{10,}", "***", s)
    return s