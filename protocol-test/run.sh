#!/bin/bash

lower_limit_in_sec=5
upper_limit_in_sec=15

generate_random_sleep() {
  echo $((RANDOM % (upper_limit_in_sec - lower_limit_in_sec + 1) + lower_limit_in_sec))
}

while true; do
  python3 my_script.py

  # Generate random sleep time
  sleep_time=$(generate_random_sleep)
  
  echo "Sleeping for $sleep_time seconds..."
  sleep $sleep_time
done