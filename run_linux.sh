#!/bin/bash
set -e

# --- CONFIGURATION ---
INPUT_DIR="input"
OUTPUT_DIR="output"
TOOLS_DIR="$PWD/tools"

# Create directories if they don't exist
mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Check if input directory is empty (warn user)
if [ -z "$(ls -A "$INPUT_DIR")" ]; then
   echo "-------------------------------------------------------------------------------"
   echo "Folders created!"
   echo "Please put your MKV files inside the '$INPUT_DIR' folder and run this script again."
   echo "-------------------------------------------------------------------------------"
   exit 0
fi

# --- STEP 0: SETUP ---
# Create marker for tag.py (must be in tools/)
touch "tools/sh-used-run_linux.sh.txt"

# Export absolute path to tools so it works when we change directories
export PATH="$TOOLS_DIR:$PATH"

# --- STEP 1: WORKER COUNT CHECK ---
# We run this from root first to ensure config is saved in tools/
if [ -f "tools/workercount-config.txt" ]; then
    WORKER_COUNT=$(grep "workers=" "tools/workercount-config.txt" | cut -d'=' -f2)
else
    echo "-------------------------------------------------------------------------------"
    echo "First Run Detected: Calculating optimal worker count..."
    echo "-------------------------------------------------------------------------------"
    python3 "tools/workercount.py"
    if [ -f "tools/workercount-config.txt" ]; then
        WORKER_COUNT=$(grep "workers=" "tools/workercount-config.txt" | cut -d'=' -f2)
    else
        WORKER_COUNT=1
    fi
fi

# --- SWITCH TO INPUT DIRECTORY ---
echo "Switching context to '$INPUT_DIR'..."
cd "$INPUT_DIR" || exit 1

# --- STEP 2: RENAMING ---
echo "Starting Renaming Process..."
# Call tool via parent directory reference
python3 "../tools/rename.py"

# --- STEP 3: PYTHON AUTOMATION ---
echo "Starting Auto-Boost-Av1an with $WORKER_COUNT final-pass workers..."

# Enable nullglob so loop doesn't run if no files match
shopt -s nullglob

for f in *-source.mkv; do
    echo ""
    echo "-------------------------------------------------------------------------------"
    echo "Detecting scenes for \"$f\"..."
    echo "-------------------------------------------------------------------------------"
    
    filename=$(basename -- "$f")
    filename_no_ext="${filename%.*}"
    
    # Run Scene Detect (Tool located in ../tools)
    python3 "../tools/Progressive-Scene-Detection.py" -i "$f" -o "${filename_no_ext}_scenedetect.json"

    echo ""
    echo "-------------------------------------------------------------------------------"
    echo "Processing \"$f\"..."
    echo "-------------------------------------------------------------------------------"
    
    # Run Main Script (Located in parent directory ..)
    python3 "../Auto-Boost-Av1an.py" -i "$f" --scenes "${filename_no_ext}_scenedetect.json" --photon-noise 2 --resume --workers "$WORKER_COUNT" --verbose --fast-params "--psy-rd 0.6" --final-params "--psy-rd 0.6 --lp 3"
done

# --- STEP 4: MUXING ---
echo "Starting Muxing Process..."
python3 "../tools/mux.py"

# --- STEP 5: TAGGING ---
# We must return to root for tagging so it finds the tools marker correctly
cd ..
echo "Tagging output files..."
# tag.py will recursively scan directories, so it will find the files in input/
python3 "tools/tag.py"

# --- STEP 6: CLEANUP & MOVE ---
# Go back to input to clean up
cd "$INPUT_DIR" || exit 1

echo "Cleaning up temporary files..."
python3 "../tools/cleanup.py"

echo "Moving finished files to '$OUTPUT_DIR'..."
# Move all *-output.mkv files to the output directory
mv *-output.mkv "../$OUTPUT_DIR/" 2>/dev/null || echo "No output files found to move."

echo "All tasks finished."