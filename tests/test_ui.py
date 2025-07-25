"""Unit tests for ui.py."""

from unittest.mock import MagicMock, patch

import pandas as pd

from mlops_iris.ui import display_result, input_form, set_page, show_footer


@patch("mlops_iris.ui.st.slider")
@patch("mlops_iris.ui.st.columns")
@patch("mlops_iris.ui.st.markdown")
def test_input_form(mock_markdown, mock_columns, mock_slider) -> None:
    """Test the input_form function to ensure it returns the expected DataFrame based on slider inputs.

    :param mock_markdown: Mocked Markdown display.
    :param mock_columns: Mocked column layout from Streamlit.
    :param mock_slider: Mocked slider values for input features.
    """
    # Mock Streamlit slider return values
    mock_columns.return_value = (MagicMock(), MagicMock())
    mock_slider.side_effect = [5.1, 1.4, 3.5, 0.2]

    df = input_form()

    expected = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
    )
    pd.testing.assert_frame_equal(df, expected)


@patch("mlops_iris.ui.st.markdown")
def test_display_result(mock_markdown) -> None:
    """Test that display_result renders the correct prediction with Markdown.

    :param mock_markdown: Mocked Markdown display function.
    """
    display_result("setosa")
    mock_markdown.assert_called_once()
    assert "setosa" in mock_markdown.call_args[0][0]


@patch("mlops_iris.ui.st.markdown")
def test_show_footer(mock_markdown) -> None:
    """Test that the footer is rendered correctly with author attribution.

    :param mock_markdown: Mocked Markdown function for footer.
    """
    show_footer()
    mock_markdown.assert_called_once()
    assert "Mehmet Acikgoz" in mock_markdown.call_args[0][0]


@patch("mlops_iris.ui.st.set_page_config")
def test_set_page(mock_set_config) -> None:
    """Test that the page configuration is set correctly in Streamlit.

    :param mock_set_config: Mocked Streamlit set_page_config function.
    """
    set_page()
    mock_set_config.assert_called_once()
