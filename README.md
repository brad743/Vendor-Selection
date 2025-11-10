Create local repo, add README, push main: mkdir Vendor-Selection cd Vendor-Selection git init echo "# Vendor-Selection" > README.md git add README.md git commit -m "initial commit" git remote add origin git@github.com:brad743/Vendor-Selection.git git push -u origin main

Create the new branch and add the script: git checkout -b add-improved-matcher mkdir -p scripts

create scripts/improved_vendor_matcher.py with the content provided below
git add scripts/improved_vendor_matcher.py git commit -m "Add improved vendor-requirement matching script" git push -u origin add-improved-matcher
