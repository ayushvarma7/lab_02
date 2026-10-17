# File: main.py
from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param
from airflow.task.trigger_rule import TriggerRule

from src.model_development import (
    load_data,
    data_preprocessing,
    separate_data_outputs,
    build_model,
    load_model,
)

def quality_gate(ti, params):
    """
    Branching function to check model accuracy against a threshold.
    If accuracy >= threshold, proceed to send acceptance email and trigger Flask DAG.
    Otherwise, send the rejection email.
    """

    metrics = ti.xcom_pull(task_ids="load_model_task")
    threshold = float(params["accuracy_threshold"])
    print(f"Model accuracy: {metrics.get('accuracy',0)} vs threshold={threshold}" )
    if metrics["accuracy"]>= threshold:
        return "send_accepted_email"
    else:
        return "send_rejection_email"

# ---------- Default args ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

# ---------- DAG ----------
dag = DAG(
    dag_id="Training_Model_Pipeline",
    default_args=default_args,
    description="Airflow-Lab2 DAG Description",
    schedule="@daily",
    catchup=False,
    tags=["ayush_varma", "lab2"],
    owner_links={"Ayush Varma": "https://github.com/ayushvarma7"},
    max_active_runs=1,
    params = {
        "accuracy_threshold": Param(0.9, type="number", minimum=0.0, maximum=1.0,
        description= "Minimum threshold for model accuracy to trigger the Flask API DAG")
    }
)

# ---------- Tasks ----------
owner_task = BashOperator(
    task_id="task_using_linked_owner",
    bash_command="echo 1",
    owner="Ayush Varma",
    dag=dag,
)

load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id="data_preprocessing_task",
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

separate_data_outputs_task = PythonOperator(
    task_id="separate_data_outputs_task",
    python_callable=separate_data_outputs,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

build_save_model_task = PythonOperator(
    task_id="build_save_model_task",
    python_callable=build_model,
    op_args=[separate_data_outputs_task.output, "model.sav"],
    dag=dag,
)

load_model_task = PythonOperator(
    task_id="load_model_task",
    python_callable=load_model,
    op_args=[separate_data_outputs_task.output, "model.sav"],
    dag=dag,
)


# Adding a BranchPythonOperator to check model accuracy against a threshold
quality_gate_task = BranchPythonOperator(
    task_id="quality_gate_task",
    python_callable=quality_gate,
    dag=dag,
)

send_accepted_email = EmailOperator(
    task_id="send_accepted_email",
    to="varma.ay@northeastern.edu",
    subject="Notification from Airflow",
    html_content="<p>This model accuracy is above the threshold hence the model is accepted.</p>",
    dag=dag,
)

send_rejection_email = EmailOperator(
    task_id="send_rejection_email",
    to="varma.ay@northeastern.edu",
    subject ="Model Accuracy Below Threshold",
    html_content="<p>The model accuracy is below the defined threshold and will not be accepted.</p>",
    dag=dag,
)

# Fire-and-forget trigger so this DAG can finish cleanly.
trigger_dag_task = TriggerDagRunOperator(
    task_id="my_trigger_task",
    trigger_dag_id="WebService_Flask",
    conf={"message": "Data from upstream DAG"},
    reset_dag_run=False,
    wait_for_completion=False,          # don't block
    trigger_rule=TriggerRule.ALL_DONE,  # still run even if something upstream fails
    dag=dag,
)

# ---------- Dependencies ----------
owner_task >> load_data_task >> data_preprocessing_task >> \
    separate_data_outputs_task >> build_save_model_task >> \
    load_model_task >> quality_gate_task >> [send_accepted_email, send_rejection_email] >> trigger_dag_task
