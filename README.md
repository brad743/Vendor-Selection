Vendor-Selection
Commit directly to main (or create default branch) This will create the default branch so I can push a branch and file.
Initialize and push from your machine (git) Run these commands locally (replace with HTTPS if you prefer):
Create local repo, add README, push main: mkdir Vendor-Selection cd Vendor-Selection git init echo "# Vendor-Selection" > README.md git add README.md git commit -m "initial commit" git remote add origin git@github.com:brad743/Vendor-Selection.git git push -u origin main

Create the new branch and add the script: git checkout -b add-improved-matcher mkdir -p scripts

create scripts/improved_vendor_matcher.py with the content provided below
git add scripts/improved_vendor_matcher.py git commit -m "Add improved vendor-requirement matching script" git push -u origin add-improved-matcher

Use GitHub CLI (if you have gh)
Create an initial commit via gh or web UI, then: gh repo clone brad743/Vendor-Selection cd Vendor-Selection git checkout -b add-improved-matcher mkdir -p scripts
create file scripts/improved_vendor_matcher.py with the content below
git add scripts/improved_vendor_matcher.py git commit -m "Add improved vendor-requirement matching script" git push --set-upstream origin add-improved-matcher gh pr create --title "Add improved vendor matcher" --body "Adds TF-IDF + fuzzy matching script for vendor selection" --base main
The script file to add (You can copy the whole file and create scripts/improved_vendor_matcher.py locally or in the GitHub web editor.)
