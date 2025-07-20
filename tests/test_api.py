"""Unit tests for api.py."""

from unittest.mock import patch

import pandas as pd
import pytest

from mlops_iris.api import call_serving_endpoint, get_databricks_token

# ---------- Tests for get_databricks_token ----------


@patch("mlops_iris.api.requests.post")
def test_get_databricks_token_success(mock_post) -> None:
    """Test successful authentication to Databricks and token retrieval.

    :param mock_post: Mocked requests.post function.
    """
    mock_response = mock_post.return_value
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"access_token": "mocked_token"}

    token = get_databricks_token(
        host="https://test-host", client_id="test_client_id", client_secret="test_client_secret"
    )

    assert token == "mocked_token"
    mock_post.assert_called_once()
    assert mock_post.call_args[1]["auth"].username == "test_client_id"


@patch.dict(
    "os.environ",
    {"DATABRICKS_CLIENT_ID": "env_client_id", "DATABRICKS_CLIENT_SECRET": "env_client_secret"},
)
@patch("mlops_iris.api.requests.post")
def test_get_databricks_token_with_env(mock_post) -> None:
    """Test token retrieval when client_id and client_secret are taken from environment variables.

    :param mock_post: Mocked requests.post function.
    """
    mock_response = mock_post.return_value
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"access_token": "env_token"}

    token = get_databricks_token(host="https://test-host")
    assert token == "env_token"
    mock_post.assert_called_once()


@patch("mlops_iris.api.requests.post")
def test_get_databricks_token_failure(mock_post) -> None:
    """Test that get_databricks_token raises an exception on failed request.

    :param mock_post: Mocked requests.post with an error side effect.
    """
    mock_post.return_value.raise_for_status.side_effect = Exception("HTTP error")

    with pytest.raises(Exception, match="HTTP error"):
        get_databricks_token("https://test-host", "id", "secret")


# ---------- Tests for call_serving_endpoint ----------


@patch("mlops_iris.api.requests.post")
def test_call_serving_endpoint_success(mock_post) -> None:
    """Test a successful call to the model serving endpoint with input DataFrame.

    :param mock_post: Mocked requests.post function.
    """
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"])

    mock_response = mock_post.return_value
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"predictions": ["setosa"]}

    result = call_serving_endpoint("https://endpoint", "fake_token", df)

    assert result == {"predictions": ["setosa"]}
    mock_post.assert_called_once()
    assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer fake_token"


@patch("mlops_iris.api.requests.post")
def test_call_serving_endpoint_failure(mock_post) -> None:
    """Test that call_serving_endpoint raises an exception when the endpoint responds with an error.

    :param mock_post: The mocked requests.post or similar function to simulate error behavior.
    """
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"])

    mock_post.return_value.raise_for_status.side_effect = Exception("Bad Request")

    with pytest.raises(Exception, match="Bad Request"):
        call_serving_endpoint("https://endpoint", "fake_token", df)
