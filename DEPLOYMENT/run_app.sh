# # chmod +x script_name.sh

# #!/bin/bash

# # Open the default web browser to access http://localhost:8501
# # Note: This script assumes that the application is running at http://localhost:8501

# docker build -t my_project_image .
# docker run -p 8501:8501 my_project_image

# # Wait for the Streamlit app to start (adjust the sleep duration as needed)
# sleep 5

# # Check if the 'xdg-open' command is available (works on Linux systems)
# if command -v xdg-open &>/dev/null; then
#   xdg-open http://localhost:8501
# # Check if the 'open' command is available (works on macOS)
# elif command -v open &>/dev/null; then
#   open http://localhost:8501
# # If neither 'xdg-open' nor 'open' is available, print an error message
# else
#   echo "Unable to open web browser. Please navigate to http://localhost:8501 manually."
# fi

docker build -t streamlit-app .

docker run -p 8501:8501 streamlit-app



