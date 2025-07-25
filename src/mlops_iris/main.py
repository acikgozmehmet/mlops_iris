"""main app module."""

import os

import requests
import streamlit as st

if os.getenv("DATABRICKS_WORKSPACE_ID", None):
    from api import (  # noqa: F401  # ignore unused import warnings
        call_serving_endpoint,
        get_databricks_token,
    )
    from config import config
    from ui import display_result, inject_css, input_form, set_page, show_footer
else:
    from dotenv import load_dotenv

    load_dotenv(override=True)
    from mlops_iris.api import (  # noqa: F401  # ignore unused import warnings
        call_serving_endpoint,
        get_databricks_token,
    )
    from mlops_iris.config import config
    from mlops_iris.ui import (
        display_result,
        inject_css,
        input_form,
        set_page,
        show_footer,
    )


def main() -> None:
    """Run the Iris Species Predictor Streamlit app.

    This function handles CSS injection, user input, prediction requests, and result display.
    """
    inject_css()
    input_df = input_form()
    if st.button("🔮 Predict Species"):
        try:
            # Uncomment the following line to run the app in deployment
            token = get_databricks_token(host=config.HOST)
            # Uncomment the following line to run the app locally using a stored access token.
            # token = os.getenv("ACCESS_TOKEN")
            response = call_serving_endpoint(
                serving_endpoint=config.SERVING_ENDPOINT, token=token, input_df=input_df
            )
            predicted_species = response["predictions"][0]
            display_result(predicted_species)
        except requests.exceptions.HTTPError as e:
            st.error(f"API Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
    show_footer()


if __name__ == "__main__":
    # This must be the first Streamlit command  whenever the app runs.
    set_page()
    main()
