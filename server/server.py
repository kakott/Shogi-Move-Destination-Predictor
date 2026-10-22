if __package__:
    from .app import app as app
else:
    from app import app as app

__all__ = ["app"]
