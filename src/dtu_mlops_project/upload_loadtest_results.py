"""
This module is a utility script for uploading load testing results to W&B
and presenting the results in a neat report.
"""

import csv
import sys
from datetime import date
from typing import Annotated

import pandas
from loguru import logger
import typer
import wandb
import wandb_workspaces.reports.v2 as wr


app = typer.Typer()
run = wandb.init(project="dtu_mlops_project", job_type="add-dataset")
today = date.today()


@app.command()
def upload_results(
    result_stats: Annotated[
        str, typer.Option("--result-stats", help="Path to the 'result_stats.csv' file")
    ] = "./result_stats.csv",
    result_failures: Annotated[
        str, typer.Option("--result-failures", help="Path to the 'result_failures.csv' file")
    ] = "./result_failures.csv",
    result_exceptions: Annotated[
        str, typer.Option("--result-exceptions", help="Path to the 'result_exceptions.csv' file")
    ] = "./result_exceptions.csv",
    result_stats_history: Annotated[
        str, typer.Option("--result-stats-history", help="Path to the 'result_stats_history.csv' file")
    ] = "./result_stats_history.csv",
) -> None:
    """
    Uploads the 4 .csv files containing the data from the results of the
    locust load tests. The files will be uploaded to W&B as a single
    artifact under the name 'locust-loadtest-<date-of-today>'.

    :param result_stats: Path to the 'result_stats.csv' file
    :param result_failures: Path to the 'result_failures.csv' file
    :param result_exceptions: Path to the 'result_exceptions.csv' file
    :param result_stats_history: Path to the 'result_stats_history.csv' file
    """
    loadtest_result_artifact = wandb.Artifact(name=f"locust-loadtest-{today}", type="dataset")

    loadtest_result_artifact.add_file(local_path=result_stats, name="result_stats")
    loadtest_result_artifact.add_file(local_path=result_failures, name="result_failures")
    loadtest_result_artifact.add_file(local_path=result_exceptions, name="result_exceptions")
    loadtest_result_artifact.add_file(local_path=result_stats_history, name="result_stats_history")

    loadtest_result_artifact.save()

    logger.info(f"Successfully uploaded artifact: 'locust-loadtest-{today}' to W&B.")


@app.command()
def create_report() -> None:
    """
    Creates a report in W&B covering the latest load test performed
    using locust.
    """
    loadtest_report = wr.Report(
        title=f"Locust load test results {today}",
        description=f"Details about the locust load test result from {today}",
    )


if __name__ == "__main__":
    app()
