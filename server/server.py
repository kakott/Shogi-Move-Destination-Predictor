if __package__:
    from .app.main import app as app
else:
    from app.main import app as app

__all__ = ["app"]
