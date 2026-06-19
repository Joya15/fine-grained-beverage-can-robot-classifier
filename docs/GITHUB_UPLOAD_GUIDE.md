# GitHub Upload Guide

From this folder:

```bash
git init
git add .
git commit -m "Add fine-grained beverage can robot classifier"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Recommended repository name:

```text
fine-grained-beverage-can-robot-classifier
```

Do not commit local datasets or model checkpoints. If you want to share trained weights, upload them through GitHub Releases, Google Drive, or another artifact store and link them from the README.

