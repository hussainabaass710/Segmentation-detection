# Setting Up Remote Repository

## Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Repository name: `Segmentation-detection` (or `segmentation-detection`)
3. Description: "Point cloud semantic segmentation using PointNet++"
4. Choose visibility (public/private)
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)

## Connect Local Repository to GitHub

After creating the repository on GitHub, run:

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/Segmentation-detection.git

# Or if using SSH:
git remote add origin git@github.com:YOUR_USERNAME/Segmentation-detection.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Verify Remote

```bash
git remote -v
```

You should see:
```
origin  https://github.com/YOUR_USERNAME/Segmentation-detection.git (fetch)
origin  https://github.com/YOUR_USERNAME/Segmentation-detection.git (push)
```

## Future Updates

After making changes:

```bash
git add .
git commit -m "Your commit message"
git push
```

