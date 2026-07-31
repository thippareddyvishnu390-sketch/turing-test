import pytest
from typing import Generator
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """
    Pytest fixture that provides a configured FastAPI TestClient.
    
    The scope is set to 'session' to reuse the client instance 
    across the entire test suite for efficiency.
    
    Yields:
        TestClient: An instance of the FastAPI test client.
    """
    with TestClient(app) as test_client:
        yield test_client