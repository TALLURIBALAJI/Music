import subprocess
import sys
import os
import time

# Starts Streamlit using the current Python executable, sends a blank line to stdin
# to skip the onboarding email prompt, and leaves the server running.

cmd = [sys.executable, '-m', 'streamlit', 'run', os.path.join(os.getcwd(), 'app.py')]

# Start the Streamlit process
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# Give it a moment, then send a newline to respond to onboarding prompt
time.sleep(1)
try:
    p.stdin.write(b"\n")
    p.stdin.flush()
    p.stdin.close()
except Exception:
    pass

# Save PID so it can be stopped later
with open('streamlit_helper.pid', 'w') as f:
    f.write(str(p.pid))

print(f"Streamlit started (PID {p.pid}). Open http://localhost:8501 in your browser.")
