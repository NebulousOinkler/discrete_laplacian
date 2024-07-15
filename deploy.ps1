# Path to the virtual environment
# $envPath = "C:\Users\Sharat\Sites\discrete_laplacian_env\Scripts\Activate.ps1"

# # Activate the virtual environment
# & $envPath

# # Path to the project directory
# $projectPath = "C:\Users\Sharat\Sites\discrete_laplacian"

# # Change to the project directory
# Set-Location $projectPath

# Generate the static files
# pelican content

# Copy the CNAME file to the output directory
#Copy-Item -Path "CNAME" -Destination "output\CNAME"

# Commit and push source files to the main branch
# git add .
# git commit -m "Update site content"
# git push origin main

# Deploy to GitHub Pages
ghp-import output
git push origin gh-pages --force

# Deactivate the virtual environment (if needed)
# To deactivate in PowerShell, you usually just exit the session or close the terminal
