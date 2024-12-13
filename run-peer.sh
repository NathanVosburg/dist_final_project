#!/bin/bash

# Trap cleanup function on script exit
cleanup() {
    echo "Cleaning up processes..."
    # Kill all background processes in the same process group
    kill -TERM -$$
    wait
    echo "Cleanup complete"
    exit 0
}

# Set up trap
trap cleanup EXIT SIGINT SIGTERM

# Check if a port argument is provided
if [ -z "$1" ]; then
    echo "Error: Missing 'num' argument."
    echo "Usage: $0 <NUM>"
    exit 1
fi

# Validate that the port is a valid number
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: Num must be a numeric value."
    exit 1
fi

# Assign the port argument
NUM=$1

# Define commands
GOCMD="go run modelservice_server/modelservice_server.go -port 800$NUM"
PYCMD="python peer.py ./client_data/$NUM.json 800$NUM"

# Start processes
eval $GOCMD &
GO_PID=$!
eval $PYCMD &
PY_PID=$!

wait

cleanup