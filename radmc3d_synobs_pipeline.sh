#!/bin/bash

# Function to handle cleanup and exit
cleanup() {
    echo "Script interrupted. Exiting..."
    exit 1
}
# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT

# so casa commmand can be used
source ~/.bashrc

# Define the default value for the steps to execute
execute_steps="inp radimage radvis synobs tclean visan"

# Log file path
log_file="execution.log"

# Check if the log file exists
if [ ! -f "$log_file" ]; then
    # Initialize the log file (create new if not exists)
    echo "Execution log - $(date)" > $log_file
    echo "=========================" >> $log_file
else
    # Append to the log file
    echo "Execution log - $(date)" >> $log_file
    echo "=========================" >> $log_file
fi

# Function to run a specific step
run_step() {
    step_name=$1
    echo "Running $step_name" | tee -a $log_file
    case "$step_name" in
        inp)
            python setup.py >> $log_file 2>&1
            python dusttemp_setup.py >> $log_file 2>&1
            ;;
        radimage)
            rm -f radmc3d_1300*
            rm -f radmc3d_7000*
            echo "%run plotimage.py" | ipython --matplotlib=tk >> $log_file 2>&1
            ;;
        radvis)
            conda run -n aplpy_env python aplypy.py >> $log_file 2>&1
            ;;
        synobs)
            rm -r -f band6*
            rm -r -f bandQ*        
            casa --nologger -c alma_synobs_dirtymap.py >> $log_file 2>&1
            casa --nologger -c vla_synobs_dirtymap.py >> $log_file 2>&1
            ;;
        tclean)
            rm -f synobs_1300*
            rm -f synobs_7000*
            casa --nologger -c alma_tclean.py >> $log_file 2>&1
            casa --nologger -c vla_tclean.py >> $log_file 2>&1
            ;;
        visan)
            #conda run -n aplpy_env python synobs_obs_plotting.py >> $log_file 2>&1
            conda run -n aplpy_env python analysis.py >> $log_file 2>&1
            ;;
        *)
            echo "Invalid step: $step_name" | tee -a $log_file
            ;;
    esac

    # Check the exit code of the previous command
    if [ $? -ne 0 ]; then
        echo "Error in $step_name. Stopping the process." | tee -a $log_file
        exit 1
    else
        echo "Completed $step_name" | tee -a $log_file
    fi
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dirty)
            execute_steps="inp radimage radvis synobs"
            shift
            ;;
        -c|--clean)
            execute_steps="tclean visan"
            shift
            ;;
        -ro|--radmc3donly)
            execute_steps="inp radimage"
            shift
            ;;
        *)
            # Assume it's a step name
            execute_steps="$1"
            shift
            ;;
        -*)
            echo "Invalid option: $1" >&2
            exit 1
            ;;
    esac
done

# Execute the specified step(s)
for step in $(echo "$execute_steps" | tr ',' ' '); do
    run_step "$step"
done
