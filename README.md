# Creating and Deploying a Databricks App with Asset Bundles


**Author:** Mehmet Acikgoz

Welcome to this hands-on MLOps project designed to help you build, test, and deploy machine learning applications on Databricks using the new **Asset Bundles** feature and **Databricks Free Edition**. Whether you’re a data scientist, ML engineer, or Databricks enthusiast, this repo guides you through setting up a clean, scalable environment, managing the full ML lifecycle—from training and model registry to real-time deployment—and creating an interactive Streamlit app integrated with your model.

The project emphasizes best practices such as version control with GitHub, unit testing, and CI/CD automation using GitHub Actions to ensure a smooth, production-ready workflow. Along the way, you’ll find practical code samples, clear documentation, and tips to avoid common pitfalls. Follow this series to confidently deploy Databricks ML apps and effectively manage your ML assets. Stay tuned for continuous updates and improvements!

## Part-1: Development Environment Setup
This part demonstrates a practical, step-by-step approach to **setting up a robust MLOps workflow using Databricks Asset Bundles**. The guide covers essential development environment setup, including installing **Git**, **Visual Studio Code**, and the high-performance **uv Python package manager** to streamline dependency management. It walks users through registering for the **Databricks Free Edition**, generating a **Personal Access Token (PAT)** for secure CLI authentication, and installing the **Databricks CLI** for seamless automation and resource management. With these tools configured, you can efficiently build, test, and deploy Databricks apps as asset bundles—enabling best practices like source control, CI/CD, and reproducible infrastructure for data and AI projects.

For a detailed walkthrough and practical tips, read the full article on [Medium](https://medium.com/@macikgozm/creating-and-deploying-a-databricks-app-with-asset-bundles-5ab51d552656).

## Part-2: Project Setup
This part demonstrates how to initialize and structure a Databricks MLOps application using Asset Bundles, building on a modern Python workflow for robust, reproducible development. The setup begins by scaffolding the project with the Databricks CLI, ensuring all essential files and directories are in place for a production-ready workflow. Unnecessary default folders are removed, and best-practice directories such as src/ (for core package code) and notebooks/ (for experimentation and training scripts) are added to keep the codebase modular and organized. Dependency management leverages uv for both core and optional development tools, enabling efficient installation and clear separation between production and development environments. The initial implementation features a minimal backend prediction API and a Streamlit-powered UI, providing an end-to-end workflow that can be run and tested locally from day one. This foundation supports agile, iterative development, with a focus on maintainability, testability, and future automation.

For a detailed step-by-step guide, see the accompanying Medium article on [Medium](https://medium.com/@macikgozm/creating-and-deploying-a-databricks-app-with-asset-bundles-f9395eb46f91)


## Part-3: Model Development & Deployment  
This part takes the project from setup to action, walking through the full lifecycle of developing and deploying a machine learning model in Databricks using Asset Bundles. It starts by ensuring the project is correctly configured and deployed, leveraging the Databricks CLI to validate and upload the bundle for reliable, environment-agnostic development. The workflow emphasizes organization and governance by setting up a Unity Catalog catalog and schema for structured data and model storage. Model development leverages scikit-learn pipelines for robust preprocessing and training on the classic Iris dataset, with seamless experiment tracking and model management handled via MLflow—capturing metrics, parameters, signatures, and input datasets for reproducibility. The model is then registered in Unity Catalog to enforce discoverability and version-controlled governance, before being deployed as a REST endpoint with Databricks Model Serving, complete with scalable serving options and permission management for secure API consumption. This end-to-end workflow ensures production readiness, collaborative development, and cost-efficient deployment practices, paving the way for fully integrated app experiences in subsequent installments.

For the full walkthrough and code samples, read the original [article](https://medium.com/@macikgozm/creating-and-deploying-a-databricks-app-with-asset-bundles-03382a648e90)


## Part-4: Streamlit App Development with MVC Architecture  
This part guides readers through the development of a machine learning inference application using Streamlit on Databricks, with a strong focus on code structure, testing, and maintainability. Built around the classic Iris dataset, the app follows a clean MVC architecture to decouple logic, interface, and data flow—enhancing scalability and ease of testing. The backend (`api.py`) handles authentication and model inference securely, while the frontend (`ui.py`) delivers a polished user experience via modular Streamlit components. Central control is maintained through `main.py`, which orchestrates input handling and prediction display. Emphasis is placed on engineering best practices, including pre-commit hooks (via Ruff) for code quality, thoughtful use of environment configurations, and meaningful unit tests to ensure robustness. Readers walk away with a deployable, testable app that adheres to modern MLOps workflows—ready for local testing and future CI/CD integration.

For a detailed step-by-step guide, see the accompanying Medium  [article](https://medium.com/@macikgozm/creating-and-deploying-a-databricks-app-with-asset-bundles-c16278cc9c82)


## Getting started

1. Install the Databricks CLI from https://docs.databricks.com/dev-tools/cli/databricks-cli.html

2. Authenticate to your Databricks workspace, if you have not done so already:
    ```
    $ databricks configure
    ```

3. To deploy a development copy of this project, type:
    ```
    $ databricks bundle deploy --target dev
    ```
    (Note that "dev" is the default target, so the `--target` parameter
    is optional here.)

    This deploys everything that's defined for this project.
    For example, the default template would deploy a job called
    `[dev yourname] mlops_iris_job` to your workspace.
    You can find that job by opening your workpace and clicking on **Workflows**.

4. Similarly, to deploy a production copy, type:
   ```
   $ databricks bundle deploy --target prod
   ```

   Note that the default job from the template has a schedule that runs every day
   (defined in resources/mlops_iris.job.yml). The schedule
   is paused when deploying in development mode (see
   https://docs.databricks.com/dev-tools/bundles/deployment-modes.html).

5. To run a job or pipeline, use the "run" command:
   ```
   $ databricks bundle run
   ```
6. Optionally, install developer tools such as the Databricks extension for Visual Studio Code from
   https://docs.databricks.com/dev-tools/vscode-ext.html.

7. For documentation on the Databricks asset bundles format used
   for this project, and for CI/CD configuration, see
   https://docs.databricks.com/dev-tools/bundles/index.html.


---
*Powered by Databricks Free Edition and modern MLOps principles.*
