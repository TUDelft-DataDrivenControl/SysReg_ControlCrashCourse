import os

folder = os.path.normpath("./../ipynb")
nbs = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and ".ipynb" in f]

for nb in nbs:
    os.system(f'jupyter nbconvert --to=html --output-dir=. ./../ipynb/{nb}')
