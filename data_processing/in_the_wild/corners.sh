#!/bin/bash

# Script to copy corners.py to every subfolder in a given directory

# Check if a directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <target_directory>"
    echo "Example: $0 /path/to/parent/folder"
    exit 1
fi

TARGET_DIR="$1"
SOURCE_FILE="./corners.npy"

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: $SOURCE_FILE not found in current directory"
    exit 1
fi

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist"
    exit 1
fi

echo "Copying $SOURCE_FILE to all subfolders in $TARGET_DIR"
echo "----------------------------------------"

# Counter for copied files
count=0

# Find all subdirectories and copy corners.py to each
for subfolder in "$TARGET_DIR"/*; do
    if [ -d "$subfolder" ]; then
        subfolder_name=$(basename "$subfolder")
        echo "Copying to: $subfolder_name"
        
        # Copy the file
        cp "$SOURCE_FILE" "$subfolder/"
        
        # Check if copy was successful
        if [ $? -eq 0 ]; then
            echo "  ✓ Success"
            ((count++))
        else
            echo "  ✗ Failed"
        fi
    fi
done

echo "----------------------------------------"
echo "Copied corners.py to $count subdirectories"