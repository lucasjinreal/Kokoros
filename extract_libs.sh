#!/bin/bash

# Script to extract shared libraries from ldd output and copy them to current directory
# Excludes system libs

# Check if binary is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <binary>"
    exit 1
fi

binary="$1"

# Check if binary exists
if [ ! -f "$binary" ]; then
    echo "Error: Binary '$binary' not found"
    exit 1
fi

# Extract library names and copy them using ldd
echo "Extracting libraries for $binary..."
ldd "$binary" | while IFS= read -r line; do
    # Skip lines that don't contain library information
    if [[ ! "$line" =~ "=>" ]]; then
        continue
    fi
    
    # Extract library path from the line (the first element after =>)
    lib_path=$(echo "$line" | sed 's/.*=> \(.*\) .*/\1/')
    lib_name=$(basename "$lib_path")
    
    # Skip excluded libraries
    case "$lib_name" in
        *libm.so*|*libc.so*|*libstdc++.so*|*ld-linux-x86-64.so*|*libgcc_s.so*|*libsystemd.so*|*libasyncns.so* \
        |*libssl.so*|*libcrypto.so*)
            echo "Skipping excluded library: $lib_name"
            continue
            ;;
    esac
    
    # Check if library file exists
    if [ -f "$lib_path" ]; then
        echo "Copying $lib_name -> ./$lib_name"
        cp "$lib_path" ./
    else
        echo "Warning: Library file not found: $lib_path"
    fi
done

echo "Done."