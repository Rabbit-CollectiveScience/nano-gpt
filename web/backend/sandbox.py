import traceback
import sys
import io

def execute_code(code_string):
    """
    Safely executes a mathematical Python string.
    Extracts the 'visualize_data' variable set by the user script.
    """
    # Create an isolated global dictionary
    sandbox_env = {}

    # Capture standard output (e.g. prints)
    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        # Execute the raw text as Python code
        exec(code_string, sandbox_env)
        
        # We expect the user's code to place the final tensor dimensions or arrays
        # into a specific variable called `visualize_data`
        if 'visualize_data' not in sandbox_env:
            raise ValueError("The code did not produce a 'visualize_data' variable to render.")
            
        data = sandbox_env['visualize_data']
        
        # If the data is a PyTorch tensor, we must convert it to nested native lists for JSON serialization
        import torch
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy().tolist()
            
        return {
            "output": captured_output.getvalue(),
            "visualize_data": data
        }

    except BaseException as e:
        # Return the beautiful Python traceback so the React UI can highlight the error
        error_msg = traceback.format_exc()
        raise Exception(error_msg)
    finally:
        # Always restore standard out
        sys.stdout = original_stdout
