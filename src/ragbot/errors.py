"""Errors whose message is written for the person running the bot.

Every message says what went wrong and what to do next, so the CLI and the API
can show it as is instead of a stack trace.
"""


class RagbotError(Exception):
    """Base class for all expected, user-facing failures."""


class ConfigError(RagbotError):
    """A setting is missing or invalid."""


class DocumentError(RagbotError):
    """A source document could not be read."""


class IndexStoreError(RagbotError):
    """The on-disk index is missing, unreadable or built with other settings."""


class ProviderError(RagbotError):
    """An embedding, reranking or answer provider failed."""


class GoldenSetError(RagbotError):
    """The evaluation golden set is malformed."""
