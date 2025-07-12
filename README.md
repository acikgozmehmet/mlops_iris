# Databricks MLOps with Asset Bundles

**Author:** Mehmet Acikgoz

Welcome to this hands-on MLOps project designed to help you build, test, and deploy machine learning applications on Databricks using the new **Asset Bundles** feature and **Databricks Free Edition**. Whether you’re a data scientist, ML engineer, or Databricks enthusiast, this repo guides you through setting up a clean, scalable environment, managing the full ML lifecycle—from training and model registry to real-time deployment—and creating an interactive Streamlit app integrated with your model.

The project emphasizes best practices such as version control with GitHub, unit testing, and CI/CD automation using GitHub Actions to ensure a smooth, production-ready workflow. Along the way, you’ll find practical code samples, clear documentation, and tips to avoid common pitfalls. Follow this series to confidently deploy Databricks ML apps and effectively manage your ML assets. Stay tuned for continuous updates and improvements!


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



