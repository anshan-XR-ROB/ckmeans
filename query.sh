#!/bin/sh

PID="$1"

if ps -p $PID > /dev/null; then
    echo "running"
else
    echo "finished"
fi
