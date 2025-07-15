from unittest.mock import patch, MagicMock
from mlops_iris.main import main

@patch("mlops_iris.main.display_result")
@patch("mlops_iris.main.st")
@patch("mlops_iris.main.call_serving_endpoint")
@patch("mlops_iris.main.get_databricks_token")
@patch("mlops_iris.main.input_form")
@patch("mlops_iris.main.inject_css")
@patch("mlops_iris.main.show_footer")
def test_main_success(mock_footer, mock_css, mock_form, mock_token, mock_call, mock_st, mock_display_result):
    mock_st.button.return_value = True  # simulate button press
    mock_form.return_value = MagicMock()
    mock_token.return_value = "mocked_token"
    mock_call.return_value = {"predictions": ["setosa"]}

    main()

    mock_css.assert_called_once()
    mock_form.assert_called_once()
    mock_token.assert_called_once()
    mock_call.assert_called_once()
    mock_display_result.assert_called_once_with("setosa")
    mock_footer.assert_called_once()




@patch("mlops_iris.main.st")
@patch("mlops_iris.main.call_serving_endpoint", side_effect=Exception("Something failed"))
@patch("mlops_iris.main.get_databricks_token", return_value="token")
@patch("mlops_iris.main.input_form")
@patch("mlops_iris.main.inject_css")
@patch("mlops_iris.main.show_footer")
def test_main_unexpected_exception(mock_footer, mock_css, mock_form, mock_token, mock_call, mock_st):
    mock_st.button.return_value = True
    mock_form.return_value = MagicMock()

    main()

    mock_st.error.assert_called_with("Unexpected error: Something failed")