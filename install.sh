#!/bin/bash

# Check for required dependencies
if [[ "$(uname)" == "Darwin" ]]; then
    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew is required for automatic dependency installation."
        echo "Please install Homebrew or manually install: pkg-config opus"
        exit 1
    fi

    if ! command -v pkg-config &> /dev/null; then
        echo "pkg-config not found. Installing via Homebrew..."
        brew install pkg-config
    fi

    if ! brew list opus &> /dev/null; then
        echo "opus not found. Installing via Homebrew..."
        brew install opus
    fi
fi

# Set variables
VOICES_JSON_SRC="data/voices-v1.0.bin"
VOICES_JSON_DEST="$HOME/.cache/kokoros/data.voices-v1.0.bin"
KOKO_BIN_SRC="target/release/koko"
KOKO_BIN_DEST="/usr/local/bin/koko"

# Create the destination directory if it doesn't exist
if [ ! -d "$(dirname "$VOICES_JSON_DEST")" ]; then
    echo "Creating directory: $(dirname "$VOICES_JSON_DEST")"
    mkdir -p "$(dirname "$VOICES_JSON_DEST")"
fi

# Copy voices to the cache directory
if [ -f "$VOICES_JSON_SRC" ]; then
    echo "Copying $VOICES_JSON_SRC to $VOICES_JSON_DEST"
    cp "$VOICES_JSON_SRC" "$VOICES_JSON_DEST"
else
    echo "Error: $VOICES_JSON_SRC not found. Aborting."
    exit 1
fi

# Copy koko binary to /usr/local/bin
if [ -f "$KOKO_BIN_SRC" ]; then
    echo "Copying $KOKO_BIN_SRC to $KOKO_BIN_DEST"
    sudo cp "$KOKO_BIN_SRC" "$KOKO_BIN_DEST"
else
    echo "$KOKO_BIN_SRC not found. Build for you..."
    cargo build --release
    echo "Copying $KOKO_BIN_SRC to $KOKO_BIN_DEST"
    sudo cp "$KOKO_BIN_SRC" "$KOKO_BIN_DEST"
fi

# Provide user feedback
if [ $? -eq 0 ]; then
    echo "Installation completed successfully!"
    echo "Voices configuration: $VOICES_JSON_DEST"
    echo "Executable installed at: $KOKO_BIN_DEST"
    echo '🎉 now try in terminal: koko '
else
    echo "Installation encountered an error. Please check the messages above."
    exit 1
fi
