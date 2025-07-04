import asyncio
from run_notebook import run_notebook

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Execute eda notebook and save the output
    run_notebook("notebooks/eda.ipynb",
                 save_output=True,
                 output_path="notebooks/outputs/eda_executed.ipynb")

    # Execute baseline_models notebook and save the output
    run_notebook("notebooks/baseline_models.ipynb",
                 save_output=True,
                 output_path=(
                     "notebooks/outputs/baseline_models_executed.ipynb"))

    # Execute feature_engineering notebook and save the output
    run_notebook("notebooks/feature_engineering.ipynb",
                 save_output=True,
                 output_path=(
                     "notebooks/outputs/feature_engineering_executed.ipynb"))

    # Execeute model notebook and save the output
    run_notebook("notebooks/model.ipynb",
                 save_output=True,
                 output_path=(
                     "notebooks/outputs/model_executed.ipynb"))
