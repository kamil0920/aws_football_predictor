# Define timeout in minutes
$TIMEOUT_IN_MINS = 60

# Create the directory for auto-shutdown if it doesn't exist
$autoShutdownFolder = Join-Path -Path $Env:USERPROFILE -ChildPath ".auto-shutdown"
if (-not (Test-Path $autoShutdownFolder)) {
    New-Item -ItemType Directory -Path $autoShutdownFolder
}

# Define the path for the Python script
$pythonScriptPath = Join-Path -Path $autoShutdownFolder -ChildPath "set-time-interval.py"

# Python script content
$pythonScriptContent = @"
import json
import requests

TIMEOUT = $TIMEOUT_IN_MINS
session = requests.Session()

# Getting the xsrf token first from Jupyter Server
response = session.get("http://localhost:8888/jupyter/default/tree")

# Calls the idle_checker extension's interface to set the timeout value
response = session.post("http://localhost:8888/jupyter/default/sagemaker-studio-autoshutdown/idle_checker",
                        json={"idle_time": TIMEOUT, "keep_terminals": False},
                        params={"_xsrf": response.headers['Set-Cookie'].split(";")[0].split("=")[1]})

if response.status_code == 200:
    print(f"Succeeded, idle timeout set to {TIMEOUT} minutes")
else:
    print("Error!")
    print(response.status_code)
"@

# Write the Python script content to the file
$pythonScriptContent | Out-File -FilePath $pythonScriptPath -Encoding UTF8

# Execute the Python script
python $pythonScriptPath