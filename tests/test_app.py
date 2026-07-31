from app.app import create_app


def test_app_metadata():
    application = create_app()
    assert application.title == "Turing Test Chatbot"
    assert application.version == "0.1.0"
    assert application.description
    assert "Turing Test chatbot" in application.description
