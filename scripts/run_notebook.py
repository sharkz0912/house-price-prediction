import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(notebook_path, save_output=True, output_path=None,
                 kernel_name="Python 3.10.5"):
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"Notebook file '{notebook_path}' not found.")

    try:
        with open(notebook_path, 'r', encoding='utf-8') as nb_file:
            notebook = nbformat.read(nb_file, as_version=4)

        print(f"Executing notebook: {notebook_path}")
        executor = ExecutePreprocessor(timeout=600, kernel_name=kernel_name)
        executor.preprocess(
            notebook,
            {'metadata': {'path': os.path.dirname(notebook_path)}}
        )

        if save_output:
            output_path = output_path or notebook_path.replace(
                '.ipynb', '_executed.ipynb')
            with open(output_path, 'w', encoding='utf-8') as out_file:
                nbformat.write(notebook, out_file)
            print(f"Executed notebook saved to: {output_path}")
        else:
            print("Execution complete. Output not saved.")

    except Exception as e:
        print(f"Error during notebook execution: {e}")
