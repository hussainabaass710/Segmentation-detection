# Setting Up GitHub Repository

## Option 1: Using the Setup Script (Recommended)

Run the setup script:

```bash
poetry run python setup_github_repo.py
```

The script will:
1. Prompt for repository name (default: `Segmentation-detection`)
2. Ask for description
3. Ask if you want a private repository
4. Request your GitHub Personal Access Token
5. Create the repository on GitHub
6. Connect your local repo to GitHub
7. Push all your code

### Getting a GitHub Personal Access Token

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name (e.g., "Segmentation-detection setup")
4. Select scope: **`repo`** (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again!)

## Option 2: Manual Setup

### Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `Segmentation-detection`
3. Description: "Point cloud semantic segmentation using PointNet++"
4. Choose public or private
5. **Do NOT** initialize with README, .gitignore, or license
6. Click "Create repository"

### Step 2: Connect Local Repository

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/Segmentation-detection.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Verify

```bash
git remote -v
```

You should see:
```
origin  https://github.com/YOUR_USERNAME/Segmentation-detection.git (fetch)
origin  https://github.com/YOUR_USERNAME/Segmentation-detection.git (push)
```

## Option 3: Install GitHub CLI (Alternative)

If you prefer using GitHub CLI:

```bash
# Install GitHub CLI (macOS)
brew install gh

# Authenticate
gh auth login

# Create repository
gh repo create Segmentation-detection --public --source=. --remote=origin --push
```

## Troubleshooting

### Authentication Issues

If you get authentication errors:
- Make sure your token has the `repo` scope
- Check that your token hasn't expired
- For HTTPS, you might need to use a Personal Access Token instead of password

### Repository Already Exists

If the repository name is taken:
- Choose a different name
- Or delete the existing repository on GitHub first

### Push Errors

If push fails:
- Make sure you have write access to the repository
- Check your internet connection
- Verify the remote URL is correct: `git remote -v`

