#!/bin/bash

# Activate the virtual environment
source ../discrete_laplacian_env/bin/activate

# Generate the static files
pelican content

# Copy the CNAME file to the output directory
cp CNAME output/

# Deploy to GitHub Pages
ghp-import output
git push origin gh-pages --force
