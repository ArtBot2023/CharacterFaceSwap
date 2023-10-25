import subprocess
import sys
import os

local_packages = ["./thirdparty/facexlib"]

def install_local_package(path_to_package):
    """
    Installs a local Python package using the Python interpreter executing this script.

    Args:
    - path_to_package (str): Relative or absolute path to the package directory containing setup.py.
    """
    try:
        # Ensure pip is available
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])

        # Install the local package using the current Python interpreter
        subprocess.check_call([sys.executable, "-m", "pip", "install", path_to_package])

        print(f"Successfully installed package from {path_to_package}")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing package from {path_to_package}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Specify the path to your local package
    for path_to_package in local_packages:
        # Check if the path exists
        if os.path.exists(path_to_package):
            install_local_package(path_to_package)
        else:
            print(f"Path {path_to_package} does not exist. Please check and try again.")

