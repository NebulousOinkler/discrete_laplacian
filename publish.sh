#!/bin/bash

# Activate the virtual environment
..\discrete_laplacian_env\Scripts\activate
echo 'test'
# Generate the static files
pelican content

# Copy the CNAME file to the output directory
cp CNAME output/

# Commit and push source files to the main branch
git add .
git commit -m "Update site content"
git push origin main

# Deploy to GitHub Pages
ghp-import output
git push origin gh-pages --force
